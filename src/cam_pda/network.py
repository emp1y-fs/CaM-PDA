"""Final CaM-PDA: native three-channel ViT-B and independent R/N/E residuals."""
from contextlib import contextmanager
from types import MethodType
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ._vendor.pda.depth_anything_v2 import build_backbone
from ._vendor.pda.utils import disparity2depth

def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'Nonfinite {name}; no clipping or reference-domain removal is allowed')

class LowRankResidual(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        if not 0 < rank <= width:
            raise ValueError('rank must be within [1, token width]')
        self.down = nn.Linear(width, rank, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.activation(self.down(x)))

def router_condition_softsign(condition_tokens):
    """Detached, parameter-free FP32 transform; never modifies its input.

    Only conditions are bounded. This does not assert that arbitrary learned
    router weights or arbitrary token features have globally bounded logits.
    Nonfinite input is still a failure, not silently repaired or discarded.
    """
    if not isinstance(condition_tokens, torch.Tensor) or not condition_tokens.is_floating_point():
        raise TypeError('Router conditions must be a floating tensor')
    with torch.autocast(device_type=condition_tokens.device.type, enabled=False):
        value = condition_tokens.detach().float()
        _finite(value, 'three-channel router conditions before softsign')
        bounded = value / (1. + value.abs())
        _finite(bounded, 'softsign router conditions')
    return bounded

class ThreeChannelDualResidualFFN(nn.Module):
    expert_names = ('reflective', 'nonflat', 'edge')
    def __init__(self, general_ffn, width, rank=32):
        super().__init__()
        self.general_ffn = general_ffn
        self.temperature = 0.7
        self.enable_edge_expert = True
        self.routing_mode = 'sparse_all'
        self._condition_tokens = None
        self.last_probabilities = self.last_logits = self.last_activations = None
        for name in self.expert_names:
            setattr(self, name+'_residual', LowRankResidual(width, rank))
            setattr(self, name+'_router', nn.Linear(width+3, 1))
            self.register_parameter(name+'_gate_logit', nn.Parameter(torch.tensor(math.log(.1/.9))))

    @property
    def edge_gate(self):
        return self.edge_gate_logit.sigmoid()

    def set_routing(self, mode):
        if mode not in ('sparse_all', 'general_only', 'edge_disabled', 'soft', 'hard_st'):
            raise ValueError('Unsupported routing mode')
        self.routing_mode = mode

    @property
    def reflective_gate(self):
        return self.reflective_gate_logit.sigmoid()

    @property
    def nonflat_gate(self):
        return self.nonflat_gate_logit.sigmoid()

    @contextmanager
    def condition_context(self, condition_tokens):
        previous = self._condition_tokens
        self._condition_tokens = condition_tokens
        try:
            yield
        finally:
            self._condition_tokens = previous

    def _probabilities(self, x, condition_tokens):
        if (not isinstance(condition_tokens, torch.Tensor) or condition_tokens.ndim != 3
                or condition_tokens.shape != (*x.shape[:2], 3)
                or condition_tokens.device != x.device):
            raise ValueError('Router requires token-aligned BxNx3 pooled conditions; no fourth channel')
        bounded = router_condition_softsign(condition_tokens)
        router_input = torch.cat((x.float(), bounded), dim=-1)
        logits = torch.cat([getattr(self, name + '_router')(router_input)
                            for name in self.expert_names], dim=-1).float() / self.temperature
        probabilities = logits.sigmoid()
        return logits, torch.cat((torch.zeros_like(probabilities[:, :1]), probabilities[:, 1:]), dim=1)

    def forward(self, x, condition_tokens=None):
        if x.ndim != 3 or x.shape[1] < 2:
            raise ValueError('Expected BxNxD with one CLS and at least one spatial token')
        general = self.general_ffn(x)
        mode = self.routing_mode
        if mode == 'general_only':
            self.last_probabilities = self.last_logits = self.last_activations = None
            return general
        conditions = condition_tokens if condition_tokens is not None else self._condition_tokens
        logits, probabilities = self._probabilities(x, conditions)
        activations = probabilities >= .5
        activations[:, 0] = False
        if mode in ('sparse_dual', 'edge_disabled') and self.enable_edge_expert:
            activations[..., 2] = False
        if mode.endswith('_only'):
            name = mode.removesuffix('_only')
            if name not in self.expert_names:
                raise ValueError(f'{mode} requires an actual {name} expert')
            activations = torch.zeros_like(activations)
            activations[:, 1:, self.expert_names.index(name)] = True
        self.last_logits = logits
        self.last_probabilities = probabilities
        self.last_activations = activations.detach()
        sparse = not self.training or mode in (
            'sparse_dual', 'sparse_all', 'edge_disabled', 'reflective_only', 'nonflat_only', 'edge_only')
        experts = [(getattr(self, name + '_residual'), getattr(self, name + '_gate'))
                   for name in self.expert_names]
        if sparse:
            flat_x = x.reshape(-1, x.shape[-1])
            output = general.reshape(-1, general.shape[-1])
            flat_active = activations.reshape(-1, len(self.expert_names))
            for index, (expert, gate) in enumerate(experts):
                selected = flat_active[:, index].nonzero(as_tuple=False).squeeze(1)
                if selected.numel():
                    residual = expert(flat_x.index_select(0, selected))
                    residual = residual * gate.to(dtype=residual.dtype)
                    output = output.index_add(0, selected, residual.to(dtype=output.dtype))
            return output.view_as(general)
        if mode == 'soft':
            route = probabilities
        elif mode == 'hard_st':
            route = activations.to(probabilities.dtype) + (probabilities - probabilities.detach())
        else:
            raise RuntimeError('Invalid edge training routing state')
        output = general
        for index, (expert, gate) in enumerate(experts):
            residual = expert(x) * gate.to(dtype=x.dtype)
            output = output + (route[..., index:index + 1] * residual).to(dtype=general.dtype)
        return output

def _call_block(block, tokens, condition_tokens):
    if isinstance(block.mlp, ThreeChannelDualResidualFFN):
        with block.mlp.condition_context(condition_tokens):
            return block(tokens)
    return block(tokens)

def _threechannel_intermediate(self, x, n=1, condition=None):
    """Original non-chunked loop, passing immutable conditions to checkpoint.

    Passing the pool as an explicit checkpoint argument prevents condition
    leakage when several image forwards occur before their backward passes.
    This instance-bound method does not edit the shared/official repository.
    """
    if condition is None or condition.shape[:2] != (x.shape[0], 3) or condition.shape[-2:] != x.shape[-2:]:
        raise ValueError('Actual patch input requires aligned Bx3xHxW conditions')
    patch_h, patch_w = self.patch_embed.patch_size
    grid = (x.shape[-2] // patch_h, x.shape[-1] // patch_w)
    spatial = F.adaptive_avg_pool2d(condition.detach().float(), grid).flatten(2).transpose(1, 2)
    pooled = torch.cat((torch.zeros_like(spatial[:, :1]), spatial), dim=1)
    tokens = self.prepare_tokens_with_masks(x, condition=condition)
    if tokens.shape[:2] != pooled.shape[:2]:
        raise ValueError('Actual patch/CLS token count does not match three-channel pool')
    output = []
    blocks_to_take = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
    for index, block in enumerate(self.blocks):
        def run(value, pool, current=block):
            return _call_block(current, value, pool)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            tokens = checkpoint(run, tokens, pooled, use_reentrant=False, preserve_rng_state=True)
        else:
            tokens = run(tokens, pooled)
        if index in blocks_to_take:
            output.append(tokens)
    if len(output) != len(blocks_to_take):
        raise RuntimeError('Intermediate block selection mismatch')
    return output

class CaMPDANetwork(nn.Module):
    def __init__(self, input_size=518, autocast_bfloat16=True):
        super().__init__()
        self.input_size = input_size
        self.autocast_bfloat16 = autocast_bfloat16
        self.backbone = build_backbone(depth_size='vitb', encoder_cond_dim=3)
        self.backbone.construct_aux_layers()
        for index in (6, 8, 10):
            block = self.backbone.pretrained.blocks[index]
            block.mlp = ThreeChannelDualResidualFFN(block.mlp, self.backbone.pretrained.embed_dim)
        self.backbone.pretrained._get_intermediate_layers_not_chunked = MethodType(
            _threechannel_intermediate, self.backbone.pretrained)
        self.backbone.pretrained.gradient_checkpointing = False

    def set_routing(self, mode):
        for index in (6, 8, 10):
            self.backbone.pretrained.blocks[index].mlp.set_routing(mode)

    def routing_outputs(self):
        return {str(i): {'probabilities':self.backbone.pretrained.blocks[i].mlp.last_probabilities,
                        'activations':self.backbone.pretrained.blocks[i].mlp.last_activations}
                for i in (6, 8, 10)}

    def forward(self, image_bgr_u8, condition, norm_min_m, norm_range_m):
        if (not isinstance(image_bgr_u8, torch.Tensor) or image_bgr_u8.dtype != torch.uint8
                or image_bgr_u8.ndim != 4 or image_bgr_u8.shape[1] != 3):
            raise TypeError('image_bgr_u8 must be Bx3xHxW BGR bytes')
        if (not isinstance(condition, torch.Tensor) or not condition.is_floating_point()
                or condition.shape != image_bgr_u8.shape or condition.device != image_bgr_u8.device):
            raise ValueError('condition must match the image Bx3xHxW exactly; four channels are prohibited')
        for name, value in (('condition', condition), ('norm_min_m', norm_min_m), ('norm_range_m', norm_range_m)):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.device != image_bgr_u8.device:
                raise TypeError(f'{name} must be a floating tensor on the image device')
            _finite(value, name)
        if norm_min_m.shape != (image_bgr_u8.shape[0], 1, 1, 1) or norm_range_m.shape != norm_min_m.shape:
            raise ValueError('Metric normalization must be Bx1x1x1')
        if bool((norm_range_m <= 0).any()):
            raise ValueError('Metric normalization range must be positive')
        with torch.autocast(device_type=image_bgr_u8.device.type, enabled=False):
            with torch.autocast(device_type=image_bgr_u8.device.type, dtype=torch.bfloat16,
                                enabled=image_bgr_u8.device.type == 'cuda' and self.autocast_bfloat16):
                disparity = self.backbone(image_bgr_u8, self.input_size,
                                          condition=condition.float(), device=str(image_bgr_u8.device))
            _finite(disparity, 'Conditioned MDE disparity')
            depth = disparity2depth(disparity) * norm_range_m + norm_min_m
            _finite(depth, 'metric depth')
        return depth.float()
