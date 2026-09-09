"""Initial-stage depth and router objectives."""
import math
import numpy as np
import torch
import torch.nn.functional as F
PATCH_THRESHOLDS = (.02, .12, .02)

def fixed_depth_loss(prediction, data):
    prediction = prediction.squeeze(1).float()
    valid = data['valid'].bool()
    gt = data['gt_m'].float()
    if int(valid.sum()) < 32 or not torch.isfinite(gt[valid]).all() or (gt[valid] <= 0).any():
        raise ValueError('Invalid fixed GT domain')
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError('Nonfinite training output; cannot shrink reference domain')
    return ((prediction[valid] - gt[valid]).abs() / gt[valid]).mean()

def boundary_proxies_numpy(gt, valid):
    """Return (positive, known), preserving any leading dimensions and H,W.

    All comparisons deliberately use FP32, matching the training Torch path.
    Image borders and neighborhoods containing invalid GT are unknown. Missing
    GT is not silently classified as a negative boundary label.
    """
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_)
    if gt.shape != valid.shape or gt.ndim < 2:
        raise ValueError('GT and valid must share a shape ending in H,W')
    positive = np.zeros(gt.shape, dtype=np.bool_)
    known = np.zeros(gt.shape, dtype=np.bool_)
    h, w = gt.shape[-2:]
    if h < 3 or w < 3:
        return (positive, known)
    finite_valid = valid & np.isfinite(gt) & (gt > 0)
    safe = np.where(finite_valid, gt, np.float32(0))
    center = safe[..., 1:-1, 1:-1]
    complete = np.ones(center.shape, dtype=np.bool_)
    jump = np.zeros(center.shape, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            complete &= finite_valid[..., dy:dy + h - 2, dx:dx + w - 2]
            if (dy, dx) != (1, 1):
                delta = np.abs(safe[..., dy:dy + h - 2, dx:dx + w - 2] - center)
                np.maximum(jump, delta, out=jump)
    known[..., 1:-1, 1:-1] = complete
    threshold = np.maximum(np.float32(0.01), np.float32(0.02) * center)
    positive[..., 1:-1, 1:-1] = complete & (jump > threshold)
    return (positive, known)

def boundary_domain(gt, valid):
    """Pure NumPy geometric-edge evaluation domain; no prediction is read."""
    return boundary_proxies_numpy(gt, valid)[0]

@torch.no_grad()
def boundary_proxies_torch(gt, valid):
    """Torch equivalent of boundary_proxies_numpy for B,1,H,W tensors."""
    gt = gt.float()
    valid = valid.bool()
    if gt.shape != valid.shape or gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError('Torch boundary inputs must share B,1,H,W shape')
    finite_valid = valid & torch.isfinite(gt) & (gt > 0)
    positive = torch.zeros_like(valid)
    known = torch.zeros_like(valid)
    h, w = gt.shape[-2:]
    if h < 3 or w < 3:
        return (positive, known)
    safe = torch.where(finite_valid, gt, torch.zeros_like(gt))
    center = safe[:, :, 1:-1, 1:-1]
    complete = torch.ones_like(center, dtype=torch.bool)
    jump = torch.zeros_like(center)
    for dy in range(3):
        for dx in range(3):
            complete &= finite_valid[:, :, dy:dy + h - 2, dx:dx + w - 2]
            if (dy, dx) != (1, 1):
                jump = torch.maximum(jump, (safe[:, :, dy:dy + h - 2, dx:dx + w - 2] - center).abs())
    known[:, :, 1:-1, 1:-1] = complete
    threshold = torch.maximum(torch.full_like(center, 0.01), center * 0.02)
    positive[:, :, 1:-1, 1:-1] = complete & (jump > threshold)
    return (positive, known)

@torch.no_grad()
def training_proxies(data):
    gt = data['gt_m'].float().unsqueeze(1)
    valid = data['valid'].bool().unsqueeze(1)
    sensor = data['sensor_m'].float().unsqueeze(1)
    observed = valid & torch.isfinite(sensor) & (sensor > 0)
    reflection = observed & ((sensor - gt).abs() > torch.maximum(torch.full_like(gt, 0.01), gt * 0.02))
    safe = torch.where(valid, gt, torch.zeros_like(gt))
    p = F.pad(safe, (1, 1, 1, 1), mode='replicate')
    center = p[:, :, 1:-1, 1:-1]
    neighbors = torch.stack((p[:, :, :-2, 1:-1], p[:, :, 2:, 1:-1], p[:, :, 1:-1, :-2], p[:, :, 1:-1, 2:]))
    hessian = (neighbors[0] + neighbors[1] - 2 * center).abs() + (neighbors[2] + neighbors[3] - 2 * center).abs()
    complete = F.conv2d(valid.float(), torch.ones(1, 1, 3, 3, device=gt.device), padding=1) >= 9
    jump = (neighbors - center).abs().amax(dim=0)
    geometric = valid & complete & (jump <= torch.maximum(torch.full_like(gt, 0.08), gt * 0.05))
    nonflat = geometric & (hessian >= torch.maximum(torch.full_like(gt, 0.004), gt * 0.003))
    edge, edge_known = boundary_proxies_torch(gt, valid)
    return ((reflection, nonflat, edge), (observed, geometric, edge_known))

def grid_size(tokens, height, width):
    pairs = []
    for h in range(1, int(math.sqrt(tokens)) + 1):
        if tokens % h == 0:
            pairs.extend(((h, tokens // h), (tokens // h, h)))
    if not pairs:
        raise ValueError('Invalid spatial token count')
    return min(pairs, key=lambda x: abs(x[1] / x[0] - width / height))

def routing_loss(outputs, data):
    if not outputs:
        return torch.zeros((), device=data['gt_m'].device)
    positive, known = training_proxies(data)
    terms = []
    for output in outputs.values():
        if output['logits'] is None:
            continue
        logits = output['logits'][:, 1:, :].float()
        channels = logits.shape[-1]
        if channels not in (2, 3):
            raise ValueError('Expected two or three independent gates')
        grid = grid_size(logits.shape[1], *data['gt_m'].shape[-2:])
        for channel in range(channels):
            support = F.adaptive_avg_pool2d(known[channel].float(), grid).flatten(1)
            fraction = F.adaptive_avg_pool2d(positive[channel].float(), grid).flatten(1) / support.clamp_min(1e-06)
            target = (fraction >= PATCH_THRESHOLDS[channel]).float()
            use = support >= 0.25
            if use.any():
                selected = target[use]
                npositive = selected.sum()
                weight = ((selected.numel() - npositive) / npositive.clamp_min(1)).clamp(1, 4)
                terms.append(F.binary_cross_entropy_with_logits(logits[:, :, channel][use], selected, pos_weight=weight))
    if terms:
        return torch.stack(terms).mean()
    reference = next((out['logits'] for out in outputs.values() if out['logits'] is not None), None)
    return reference.sum() * 0 if reference is not None else torch.zeros((), device=data['gt_m'].device)
