"""Geometry-stage objectives and validation eligibility."""
import torch
import torch.nn.functional as F

def loss(m, p, d, warm=False):
    pred = p[:, 0].float()
    gt = d['gt_m'].float()
    v = d['valid'].bool()
    assert torch.isfinite(pred).all() and (gt[v] > 0).all()
    dep = torch.stack([((pp[vv] - gg[vv]).abs() / gg[vv]).mean() for pp, gg, vv in zip(pred, gt, v)]).mean()
    ys = d['target'].flatten(2).transpose(1, 2)
    ks = d['known'].flatten(2).transpose(1, 2)
    blocks = []
    for b, r in m.routing_outputs().items():
        logits = r['logits'][:, 1:].float()
        assert logits.shape == ys.shape
        terms = []
        raw = F.binary_cross_entropy_with_logits(logits, ys, reduction='none')
        for j in range(len(pred)):
            role_terms = []
            for role in range(3):
                class_terms = []
                for positive in (False, True):
                    mask = ks[j, :, role] & (ys[j, :, role].bool() == positive)
                    if mask.any():
                        class_terms.append(raw[j, :, role][mask].mean())
                if class_terms:
                    role_terms.append(torch.stack(class_terms).mean())
            terms.append(torch.stack(role_terms).mean() if role_terms else logits[j].sum() * 0)
        blocks.append(torch.stack(terms).mean())
    route = torch.stack(blocks).mean()
    frame_slopes = []
    for j in range(len(pred)):
        slopes = []
        for axis in (-1, -2):
            a = [slice(None)] * 2
            b = a.copy()
            a[axis] = slice(1, None)
            b[axis] = slice(None, -1)
            a = tuple(a)
            b = tuple(b)
            mask = d['plane'][j][a] & d['plane'][j][b] & v[j][a] & v[j][b]
            if mask.any():
                slopes.append(((pred[j][a] - pred[j][b] - (gt[j][a] - gt[j][b])).abs() / ((gt[j][a] + gt[j][b]) * 0.5).clamp_min(0.01))[mask].mean())
        frame_slopes.append(torch.stack(slopes).mean() if slopes else pred[j].sum() * 0)
    plane = torch.stack(frame_slopes).mean()
    total = route if warm else dep + 0.05 * route + 0.1 * plane
    return (total, dict(depth=float(dep.detach()), router=float(route.detach()), plane_gradient=float(plane.detach()), total=float(total.detach())))

def mean_valid(values):
    a = [float(x) for x in values if x is not None]
    return sum(a) / len(a) if a else None

def eligible(candidate, baseline):
    a = candidate['selection']
    b = baseline['selection']
    return a['replay_absrel'] <= b['replay_absrel'] * 1.02 and a['nonflat_curve_recall'] >= b['nonflat_curve_recall'] - 0.05 and (a['edge_recall'] >= b['edge_recall'] - 0.05)
