"""Three native PDA conditions from a fixed accepted-anchor mask."""
import torch
from .rng import alignment_rng
from ._vendor.pda.utils import depth2disparity, disparity2depth
class InsufficientAnchorsError(ValueError):
    """The rule mask cannot support the historical minimum/original KNN."""

    def __init__(self, batch, count, required):
        self.batch = batch
        self.count = count
        self.required = required
        super().__init__(f'Batch {batch}: fixed rule mask has {count} anchors; '
                         f'requires at least {required}=max(K+1,17). '
                         'No points added; caller must record failure/fallback.')


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'Nonfinite {name}')


def _checked_inputs(data, mask, seed):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 3:
        raise TypeError('mask must be a BxHxW bool tensor')
    if any(size == 0 for size in mask.shape) or mask.device.type not in ('cpu', 'cuda'):
        raise ValueError('mask requires a nonempty CPU/CUDA BxHxW grid')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0,2**63)')
    fields = {}
    # Read only the observations and prior used to construct the conditions.
    for name in ('raw_pred_disparity', 'raw_sparse_disparity', 'sensor_m', 'sampled'):
        value = data[name]
        if not isinstance(value, torch.Tensor) or value.shape != mask.shape or value.device != mask.device:
            raise ValueError(f'{name} must match mask shape and device')
        if name == 'sampled':
            if value.dtype != torch.bool:
                raise TypeError('sampled must be bool')
            fields[name] = value.detach()
            continue
        if not value.is_floating_point():
            raise TypeError(f'{name} must be floating point')
        value = value.detach().float()
        _finite(value, name)
        fields[name] = value
    sampled = fields['sampled']
    if bool((mask & ~sampled).any()):
        raise ValueError('mask must be a subset of sampled points')
    if bool((sampled & ((fields['sensor_m'] <= 0) | (fields['raw_sparse_disparity'] <= 0))).any()):
        raise ValueError('Every sampled point requires positive sensor_m and raw_sparse_disparity')
    return fields


@torch.no_grad()
def build_from_mask(pda, data, mask, seed):
    """Build scale-aligned and KNN-prefilled conditions from fixed anchors.

    Numeric fields must be finite FP32. Sensor depth and sparse disparity
    must be positive at sampled pixels. Each item needs max(K+1, 17)
    accepted anchors; callers handle an insufficient-anchor exception.

    Returns the three-channel tensor, depth normalization and anchor masks.
    """
    if pda.args.double_global or pda.args.confidence_filter or not pda.args.normalize_depth:
        raise ValueError('Require original global+KNN, no internal confidence filter, normalization enabled')
    if pda.args.extra_condition != 'spmask':
        raise ValueError('Require original sparse-mask three-channel conditions')
    K = pda.completion.K
    if isinstance(K, bool) or not isinstance(K, int) or K < 1 or pda.args.K != K:
        raise ValueError('Original KNN K must be a consistent positive integer')
    if pda.training or pda.model.training or any(parameter.requires_grad for parameter in pda.parameters()):
        raise ValueError('PDA must already be frozen and in eval mode; this function never changes it')
    fields = _checked_inputs(data, mask, seed)
    trusted = mask.detach().clone()
    minimum_count = max(K + 1, 17)
    for batch, count in enumerate(trusted.flatten(1).sum(1).tolist()):
        if count < minimum_count:
            raise InsufficientAnchorsError(batch, count, minimum_count)
    with torch.autocast(device_type=mask.device.type, enabled=False):
        with alignment_rng(mask.device, seed):
            global_prediction = pda.completion.ss_completer(
                sparse_disparities=fields['raw_sparse_disparity'],
                pred_disparities=fields['raw_pred_disparity'], sparse_masks=trusted)
            completed_prediction = pda.completion.kss_completer(
                sparse_disparities=fields['raw_sparse_disparity'],
                pred_disparities=fields['raw_pred_disparity'], sparse_masks=trusted,
                complete_masks=~trusted, K=K)
        for name, value in (('original global prediction', global_prediction),
                            ('original KNN prediction', completed_prediction)):
            if value.shape != mask.shape or value.device != mask.device:
                raise ValueError(f'{name} shape/device changed')
            _finite(value, name)
        if not torch.equal(trusted, mask):
            raise RuntimeError('Original solver unexpectedly changed the supplied fixed mask')
        trusted4 = trusted.unsqueeze(1)
        minimum, span = pda.zero_one_normalize(fields['sensor_m'].unsqueeze(1), trusted4, affine_only=True)
        _finite(minimum, 'original accepted minimum')
        _finite(span, 'original accepted normalization range')
        if bool((span <= 0).any()):
            raise ValueError('Original normalization range must be positive')
        global_depth = (disparity2depth(global_prediction.unsqueeze(1)) - minimum) / span
        completed_depth = (disparity2depth(completed_prediction.unsqueeze(1)) - minimum) / span
        _finite(global_depth, 'original normalized global depth')
        _finite(completed_depth, 'original normalized KNN depth')
        condition = torch.cat((trusted4, depth2disparity(global_depth), depth2disparity(completed_depth)), dim=1)
        _finite(condition, 'original three-channel condition')
    return dict(condition=condition.detach(), norm_min_m=minimum.detach(), norm_range_m=span.detach(),
                trusted_mask=trusted4.detach(), fallback_count=0, fixed_mask_unchanged=True)
