# Adapted from PDA / Marigold. See licenses/README.md in the distribution.
import torch
import numpy as np
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
