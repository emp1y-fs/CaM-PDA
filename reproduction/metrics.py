"""FP64 per-frame depth metrics on a fixed reference mask."""
import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError(message)

def depth_metrics(prediction, gt, mask):
    require(prediction.shape == gt.shape == mask.shape, 'Metric grid differs')
    count = int(mask.sum())
    if count < 32:
        return None
    require(np.all(np.isfinite(gt[mask]) & (gt[mask] > 0)), 'Invalid fixed reference')
    pred, ref = (prediction[mask].astype(np.float64), gt[mask].astype(np.float64))
    if not np.isfinite(pred).all():
        return dict(pixels=count, finite=False, nonfinite=int((~np.isfinite(pred)).sum()), AbsRel=None, MAE_m=None, RMSE_m=None)
    error = pred - ref
    return dict(pixels=count, finite=True, nonpositive=int((pred <= 0).sum()), AbsRel=float(np.mean(np.abs(error) / ref)), MAE_m=float(np.mean(np.abs(error))), RMSE_m=float(np.sqrt(np.mean(error ** 2))))
