"""Strict aligned RGB-D API contracts; arrays use metres and true RGB."""
from dataclasses import dataclass
from typing import Any
import hashlib
import numpy as np
def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()

def prepare_inputs(
    rgb_u8: np.ndarray,
    sensor_m: np.ndarray,
    *,
    seed: int = 0,
    maximum_samples: int = 50_000,
    sampled_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Check, copy and deterministically sample real aligned observations.

    Real-frame sampling uses NumPy PCG64 without replacement, over positive
    original observations only. It is recorded as a deployment input policy,
    not as the evaluator's already-frozen sample mask. To reproduce an external
    test exactly, supply its recorded sampled_mask AND its original seed.
    No resizing, RGB/BGR swap, depth-unit guess, outlier removal or depth fit is
    performed here. Invalid depth pixels must be represented by zero by the
    calling loader; NaN/Inf/negative values are rejected rather than repaired.
    """
    if not isinstance(rgb_u8, np.ndarray) or rgb_u8.dtype != np.uint8:
        raise TypeError("RGB must be an HWC true-RGB uint8 numpy array")
    if not isinstance(sensor_m, np.ndarray) or sensor_m.dtype != np.float32:
        raise TypeError("sensor_m must be an HW float32 array in metres")
    if sensor_m.ndim != 2 or min(sensor_m.shape, default=0) <= 0:
        raise ValueError("sensor_m must have a nonempty HW grid")
    if rgb_u8.shape != (*sensor_m.shape, 3):
        raise ValueError("RGB and depth must already be aligned on the same pixel grid")
    if not np.isfinite(sensor_m).all() or np.any(sensor_m < 0):
        raise ValueError("Depth must be finite and nonnegative; use zero for invalid pixels")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(maximum_samples) is not int or maximum_samples < 17:
        raise ValueError("maximum_samples must be an integer of at least 17")
    valid = sensor_m > 0
    available = np.flatnonzero(valid)
    if available.size < 17:
        raise ValueError("At least 17 valid observed sensor points are required; no synthetic top-up")
    if sampled_mask is not None:
        if (not isinstance(sampled_mask, np.ndarray) or sampled_mask.dtype != np.bool_
                or sampled_mask.shape != sensor_m.shape):
            raise TypeError("sampled_mask must be a bool HW array matching sensor_m")
        mask = sampled_mask.copy()
        if np.any(mask & ~valid) or int(mask.sum()) < 17:
            raise ValueError("sampled_mask must retain at least 17 positive original observations")
        sampling_policy = "caller_supplied_fixed_mask_no_resampling"
    else:
        indices = available
        if available.size > maximum_samples:
            indices = np.random.default_rng(seed).choice(available, maximum_samples, replace=False)
        mask = np.zeros(sensor_m.shape, dtype=np.bool_)
        mask.reshape(-1)[indices] = True
        sampling_policy = "deployment_numpy_PCG64_uniform_without_replacement"
    rgb_chw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1)).copy()
    sensor = np.ascontiguousarray(sensor_m).copy()
    metadata = {
        "input_rgb_convention": "HWC true RGB uint8; frozen adapter reverses to BGR exactly once",
        "sensor_unit": "m", "height": int(sensor.shape[0]), "width": int(sensor.shape[1]),
        "input_resized": False, "seed": seed, "maximum_samples": maximum_samples,
        "sampling_policy": sampling_policy, "valid_sensor_count": int(available.size),
        "sampled_count": int(mask.sum()), "input_rgb_chw_sha256": _array_sha(rgb_chw),
        "input_sensor_sha256": _array_sha(sensor), "sampled_mask_sha256": _array_sha(mask),
    }
    return rgb_chw, sensor, mask, metadata

@dataclass(frozen=True)
class DepthResult:
    """CPU arrays; depth and condition retain original precision and resolution."""

    depth_m: np.ndarray
    sampled: np.ndarray
    accepted: np.ndarray
    raw_accepted: np.ndarray
    condition: np.ndarray  # 3 x H x W: accepted mask / global / prefilled disparity
    norm_min_m: float
    norm_range_m: float
    metadata: dict[str, Any]
