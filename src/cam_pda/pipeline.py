"""Portable inference for the retained CaM-PDA checkpoint.

No experiment folders, original initialization checkpoint, training labels,
platform-specific paths, or custom compiled operators are required.
"""
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import threading
import time

import numpy as np
import torch

from .confidence import local_geometry_confidence
from .conditions import build_from_mask
from .network import CaMPDANetwork
from .rng import alignment_rng
from .types import DepthResult, prepare_inputs, _array_sha
from .weights import resolve_weights
from ._vendor.pda.completion import DepthCompletion
from ._vendor.pda.utils import depth2disparity

RULE = dict(confidence_radius=11, confidence_k=16, confidence_min_neighbors=8,
            confidence_chunk_size=4096, confidence_max_rel_error=.35,
            confidence_min_cosine=.75, confidence_min_keep=17,
            confidence_noise_floor_rel=.002, confidence_flat_rel=.004,
            confidence_smooth_rel=.020, confidence_relaxed_max_error=1.,
            confidence_relaxed_min_cosine=0., confidence_global_max_robust_z=6.)
SOURCE_CHECKPOINT = "4f406c8778dd0de1b3215f8540b325b5f24043f689a30126a407a9995bc6fe48"
_TORCH_POLICY_LOCK = threading.RLock()

class _ConditionEngine(torch.nn.Module):
    def __init__(self, mde_path, device):
        super().__init__()
        self.args = SimpleNamespace(double_global=False, confidence_filter=False,
                                    normalize_depth=True, extra_condition='spmask', K=5)
        self.completion = DepthCompletion(mde_path, device)
        # The condition builder only checks this frozen placeholder; it does not run a dense fine model.
        self.model = torch.nn.Identity()
        self.eval().requires_grad_(False)

    @staticmethod
    def zero_one_normalize(depth_maps, valid_masks, affine_only=True):
        minimum = depth_maps.masked_fill(~valid_masks, float('inf')).min(dim=-1).values.min(dim=-1).values
        maximum = depth_maps.masked_fill(~valid_masks, float('-inf')).max(dim=-1).values.max(dim=-1).values
        span = maximum-minimum
        span = torch.where(span == 0, torch.ones_like(span), span)
        return minimum.view(-1,1,1,1), span.view(-1,1,1,1)

class CaMPDA:
    """A reusable, thread-serialized metric depth completion pipeline.

    ``predict(rgb, depth_m)`` accepts aligned HWC RGB uint8 and HW float32 metres.
    Zero means missing depth. The output keeps the input resolution.
    CUDA uses the evaluated BF16 fine / FP32 coarse policy when BF16 is available.
    CPU uses FP32 and is intended for compatibility checks and smaller inputs.
    """
    def __init__(self, checkpoint=None, mde_checkpoint=None, *, device='auto',
                 maximum_samples=50000, cache_dir=None, allow_download=True,
                 memory_efficient=True):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('Supported devices: cpu or cuda[:index].')
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Install a CUDA PyTorch wheel or select cpu.')
        if type(maximum_samples) is not int or maximum_samples < 17:
            raise ValueError('maximum_samples must be an integer of at least 17.')
        self.maximum_samples=maximum_samples
        self.memory_efficient=memory_efficient
        self.checkpoint,self.mde_checkpoint,self.weight_manifest=resolve_weights(
            checkpoint,mde_checkpoint,cache_dir=cache_dir,allow_download=allow_download)
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        if payload.get('format')!='cam_pda.inference.v1' or payload.get('source_checkpoint_sha256')!=SOURCE_CHECKPOINT:
            raise ValueError('This package requires the retained CaM-PDA inference checkpoint.')
        expected={'backbone':'vitb','blocks':[6,8,10],'rank':32,'conditions':3,
                  'experts':['reflective','nonflat','edge'],'temperature':0.7}
        if payload.get('architecture')!=expected:
            raise ValueError('Checkpoint architecture does not match the public CaM-PDA interface.')
        bf16=self.device.type=='cuda' and torch.cuda.is_bf16_supported()
        self.network=CaMPDANetwork(autocast_bfloat16=bf16)
        self.network.load_state_dict(payload['state_dict'],strict=True)
        del payload
        self.network.eval().requires_grad_(False)
        self.network.set_routing('sparse_all')
        self.condition_engine=_ConditionEngine(self.mde_checkpoint,self.device)
        self.last_routing={}
        self._lock=threading.RLock()

    def _accepted(self,data,seed):
        state=SimpleNamespace(args=SimpleNamespace(**RULE),K=5,device=self.device)
        failure=None
        try:
            with alignment_rng(self.device,seed):
                _,raw,_=local_geometry_confidence(state,data['raw_sparse_disparity'],data['raw_pred_disparity'],data['sampled'])
        except RuntimeError as error:
            if 'retained too few prior points' not in str(error) and 'quantile() input tensor' not in str(error):
                raise
            trace,local=error.__traceback__,None
            while trace:
                if trace.tb_frame.f_code.co_name=='local_geometry_confidence':
                    local=trace.tb_frame.f_locals
                trace=trace.tb_next
            if local is None or 'trusted' not in local:
                raise
            raw=local['trusted'].detach().clone()
            failure=str(error)
            del local,trace
        if raw.dtype!=torch.bool or raw.shape!=data['sampled'].shape or (raw&~data['sampled']).any():
            raise RuntimeError('Balanced screening changed the sampling domain.')
        if int(raw.sum())<17 and failure is None:
            failure='historical accepted count below 17; all-accept fallback'
        return raw,data['sampled'].clone() if failure else raw.clone(),failure

    def predict(self,rgb_u8,depth_m,*,seed=0,sampled_mask=None):
        """Return depth/masks/three conditions and provenance; accepts no GT or ROI."""
        rgb,sensor,sampled,metadata=prepare_inputs(rgb_u8,depth_m,seed=seed,
            maximum_samples=self.maximum_samples,sampled_mask=sampled_mask)
        with self._lock,_TORCH_POLICY_LOCK,torch.inference_mode():
            started=time.perf_counter()
            previous_tf32=torch.backends.cuda.matmul.allow_tf32
            previous_cudnn=torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=True
                if self.memory_efficient and self.device.type=='cuda':
                    self.network.cpu()
                    gc.collect()
                    torch.cuda.empty_cache()
                self.condition_engine.completion.depth_model.to(self.device)
                rgb_tensor=torch.from_numpy(rgb.copy())[None].to(self.device)
                bgr=rgb_tensor[:,[2,1,0]].contiguous()
                sensor_tensor=torch.from_numpy(sensor.copy())[None].to(self.device)
                sampled_tensor=torch.from_numpy(sampled.copy())[None].to(self.device)
                sparse=torch.where(sampled_tensor,sensor_tensor,torch.zeros_like(sensor_tensor))
                with torch.autocast(self.device.type,enabled=False):
                    raw=self.condition_engine.completion.depth_model(bgr,518,device=str(self.device)).float().squeeze(1)
                data=dict(sensor_m=sensor_tensor,sampled=sampled_tensor,
                          raw_pred_disparity=raw,raw_sparse_disparity=depth2disparity(sparse).float())
                raw_mask,accepted,fallback=self._accepted(data,seed)
                built=build_from_mask(self.condition_engine,data,accepted,seed=seed)
                if self.memory_efficient and self.device.type=='cuda':
                    self.condition_engine.completion.depth_model.cpu()
                    gc.collect()
                    torch.cuda.empty_cache()
                self.network.to(self.device)
                torch.backends.cuda.matmul.allow_tf32=True
                depth=self.network(bgr,built['condition'],built['norm_min_m'],built['norm_range_m']).float().squeeze().cpu().numpy().copy()
                if depth.shape!=sensor.shape or not np.isfinite(depth).all() or np.any(depth<=0):
                    raise RuntimeError('The model returned invalid metric depth; no output clipping was applied.')
                conditions=built['condition'][0].float().cpu().numpy().copy()
                accepted_np=accepted.squeeze(0).cpu().numpy().copy()
                raw_np=raw_mask.squeeze(0).cpu().numpy().copy()
                self.last_routing={str(i):{k:v[:,1:].detach().cpu().numpy().copy() for k,v in item.items()}
                                   for i,item in self.network.routing_outputs().items()}
                metadata.update(method='CaM-PDA',source_checkpoint_sha256=SOURCE_CHECKPOINT,
                    inference_checkpoint_sha256=self.weight_manifest['ca_m_pda']['sha256'],
                    condition_channels=3,experts=['reflective','nonflat','edge'],routing='sparse_all',
                    accepted_count=int(accepted_np.sum()),raw_accepted_count=int(raw_np.sum()),fallback_reason=fallback,
                    condition_sha256=_array_sha(conditions[None]),depth_sha256=_array_sha(depth),
                    depth_unit='m',device=str(self.device),fine_precision='bf16' if self.network.autocast_bfloat16 else 'fp32',
                    runtime_s=time.perf_counter()-started,ground_truth_used=False,output_alignment=False)
                return DepthResult(depth,sampled.copy(),accepted_np,raw_np,conditions,
                    float(built['norm_min_m']),float(built['norm_range_m']),metadata)
            finally:
                torch.backends.cuda.matmul.allow_tf32=previous_tf32
                torch.backends.cudnn.allow_tf32=previous_cudnn

    def predict_files(self,image,depth,*,depth_scale=None,seed=0):
        from .io import read_rgb,read_depth
        return self.predict(read_rgb(image),read_depth(depth,scale=depth_scale),seed=seed)

    def predict_multiview(self,rgb,depth_m,camera,references,*,seed=0):
        """Guarded two-view refinement selected from RGB-D references.

        Each reference is a dict with ``rgb``, ``raw_m`` and ``camera``.
        Pose is estimated from RGB and raw depth; unavailable support returns
        the exact single-view result. No pose or ground-truth depth is required.
        """
        with self._lock:
            return self._predict_multiview(rgb,depth_m,camera,references,seed=seed)

    def _predict_multiview(self,rgb,depth_m,camera,references,*,seed=0):
        from dataclasses import replace
        from .multiview import select_reference,fuse_reference,_json_value
        camera.validate_shape(depth_m.shape)
        started=time.perf_counter()
        single=self.predict(rgb,depth_m,seed=seed)
        target_routing=self.last_routing
        target=dict(rgb=rgb,raw_m=depth_m,K=camera.matrix,frame_id=0)
        records={}
        for index,ref in enumerate(references,1):
            ref['camera'].validate_shape(ref['raw_m'].shape)
            records[index]=dict(rgb=ref['rgb'],raw_m=ref['raw_m'],K=ref['camera'].matrix,frame_id=index)
        registration,source,attempts=select_reference(target,records,records.__getitem__)
        metadata=dict(single.metadata,mode='multiview',registration_attempts=attempts,
                      multiview_status='single_view_fallback',fused_pixels=0,
                      single_view_runtime_s=single.metadata['runtime_s'],runtime_s=time.perf_counter()-started)
        if registration is None:return replace(single,metadata=metadata)
        reference_result=self.predict(source['rgb'],source['raw_m'],seed=seed)
        self.last_routing=target_routing
        fused,reliable,_=fuse_reference(single.depth_m,depth_m,single.accepted,
            reference_result.depth_m,camera.matrix,registration['T_source_to_target'],source_K=source['K'])
        metadata.update(multiview_status='fused',fused_pixels=int(reliable.sum()),
            runtime_s=time.perf_counter()-started,
            registration=_json_value(registration),depth_sha256=_array_sha(fused),
            refinement='depth-domain reprojection with 60 mm gate; accepted raw anchors restored')
        return replace(single,depth_m=fused,metadata=metadata)
