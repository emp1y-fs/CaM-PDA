"""Reusable pose-free RGB-D registration and continuous reference-depth correction.

Public depth inputs and outputs are metres; ``T_source_to_target`` uses a
millimetre translation, matching the established benchmark geometry. Registration
uses raw sensor depth and RGB only. Continuous correction operates on predicted
depth. No trajectory or ground-truth pose is read.

ORB/PnP/ICP retains the evaluated registration policy. The continuous solver
uses the frozen 038 parameters; image bounds follow the actual array shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import cv2
import numpy as np


def _intrinsics(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("K must be a finite 3x3 camera matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("K focal lengths must be positive")
    return matrix


def _frame(frame: dict) -> dict:
    """Build a private geometry view without changing caller RGB or depth."""
    raw = np.asarray(frame["raw_m"], dtype=np.float32)
    rgb = np.asarray(frame["rgb"])
    if raw.ndim != 2 or rgb.shape != (*raw.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 HxWx3 and match raw_m HxW")
    raw = np.where(np.isfinite(raw) & (raw > 0), raw * 1000.0, 0.0)
    return dict(frame, raw=raw, K=_intrinsics(frame["K"]), keypoints=None, descriptors=None)


def _detect(frame: dict) -> None:
    if frame["keypoints"] is not None:
        return
    gray = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    detector = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=19, patchSize=31, fastThreshold=7)
    frame["keypoints"], frame["descriptors"] = detector.detectAndCompute(gray, None)


def _matches(source: np.ndarray, target: np.ndarray) -> list:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def accepted(a, b):
        return [pair[0] for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < 0.80 * pair[1].distance]

    forward, reverse = accepted(source, target), accepted(target, source)
    reversed_pairs = {(item.trainIdx, item.queryIdx) for item in reverse}
    mutual = [item for item in forward if (item.queryIdx, item.trainIdx) in reversed_pairs]
    return mutual if len(mutual) >= 20 else forward


def _points(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.column_stack(((u - K[0, 2]) * z / K[0, 0],
                            (v - K[1, 2]) * z / K[1, 1], z))


def _transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def _project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * points[:, 0] / z + K[0, 2], K[1, 1] * points[:, 1] / z + K[1, 2], z


def _pixels(points: np.ndarray, K: np.ndarray, shape: tuple[int, int]):
    u, v, z = _project(points, K)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0)
    # Test floating-point bounds before integer conversion to avoid overflow.
    finite &= (u > -1) & (u < shape[1]) & (v > -1) & (v < shape[0])
    indices = np.flatnonzero(finite)
    ui, vi = np.rint(u[indices]).astype(np.int64), np.rint(v[indices]).astype(np.int64)
    inside = (ui >= 0) & (ui < shape[1]) & (vi >= 0) & (vi < shape[0])
    return indices[inside], ui[inside], vi[inside], z


def _rotation_degrees(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))))


def _feature_error(objects: np.ndarray, images: np.ndarray, T: np.ndarray, K: np.ndarray) -> float:
    u, v, z = _project(_transform(objects, T), K)
    valid = (z > 0) & np.isfinite(u) & np.isfinite(v)
    errors = np.linalg.norm(np.column_stack((u[valid], v[valid])) - images[valid], axis=1)
    return float(np.median(errors)) if errors.size else math.inf


def _rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], T[:3, 3] = R, target_mean - R @ source_mean
    return T


def _refine(T: np.ndarray, source: dict, target: dict, objects: np.ndarray, images: np.ndarray):
    v, u = np.mgrid[0:source["raw"].shape[0]:2, 0:source["raw"].shape[1]:2]
    z = source["raw"][::2, ::2]
    valid = np.isfinite(z) & (z > 0)
    points = _points(u[valid], v[valid], z[valid], source["K"])
    initial_error = _feature_error(objects, images, T, target["K"])
    accepted, last_residual = 0, None
    for iteration in range(4):
        current = _transform(points, T)
        indices, ui, vi, projected_z = _pixels(current, target["K"], target["raw"].shape)
        if len(indices) < 100:
            break
        target_z = target["raw"][vi, ui].astype(np.float64)
        paired = (target_z > 0) & (np.abs(target_z - projected_z[indices]) < max(15.0, 40.0 - iteration * 8.0))
        indices, ui, vi, target_z = indices[paired], ui[paired], vi[paired], target_z[paired]
        if len(indices) < 100:
            break
        if len(indices) > 60000:
            take = np.linspace(0, len(indices) - 1, 60000, dtype=np.int64)
            indices, ui, vi, target_z = indices[take], ui[take], vi[take], target_z[take]
        target_points = _points(ui, vi, target_z, target["K"])
        before = np.linalg.norm(current[indices] - target_points, axis=1)
        delta = _rigid_alignment(current[indices], target_points)
        if _rotation_degrees(delta[:3, :3]) > 2.0 or np.linalg.norm(delta[:3, 3]) > 15.0:
            break
        candidate = delta @ T
        after = np.linalg.norm(_transform(points[indices], candidate) - target_points, axis=1)
        error = _feature_error(objects, images, candidate, target["K"])
        if np.median(after) < np.median(before) and error <= max(initial_error + 1.0, 1.5 * initial_error):
            T, last_residual, accepted = candidate, float(np.median(after)), accepted + 1
        else:
            break
    return T, {"accepted_icp_updates": accepted, "icp_median_3d_residual_mm": last_residual}


def _warp_mm(depth: np.ndarray, source_K: np.ndarray, target_K: np.ndarray,
             T: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    points = _points(u, v, depth[v, u].astype(np.float64), source_K)
    indices, ui, vi, z = _pixels(_transform(points, T), target_K, target_shape)
    warped = np.full(target_shape[0] * target_shape[1], np.inf, dtype=np.float64)
    np.minimum.at(warped, vi * target_shape[1] + ui, z[indices])
    warped[~np.isfinite(warped)] = 0.0
    return warped.reshape(target_shape).astype(np.float32)


def _register(source: dict, target: dict) -> dict | None:
    _detect(source)
    _detect(target)
    if source["descriptors"] is None or target["descriptors"] is None:
        return None
    matches = _matches(source["descriptors"], target["descriptors"])
    objects, images = [], []
    for match in matches:
        u, v = source["keypoints"][match.queryIdx].pt
        x, y = int(round(u)), int(round(v))
        values = source["raw"][max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        values = values[np.isfinite(values) & (values > 0)]
        if not values.size:
            continue
        z = float(np.median(values))
        objects.append(_points(np.asarray([u]), np.asarray([v]), np.asarray([z]), source["K"])[0])
        images.append(target["keypoints"][match.trainIdx].pt)
    if len(objects) < 12:
        return None
    objects, images = np.asarray(objects, dtype=np.float64), np.asarray(images, dtype=np.float64)
    # OpenCV's RANSAC seed is fixed per pair for repeatable reference selection.
    cv2.setRNGSeed(0)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objects, images, target["K"], None, iterationsCount=3000,
        reprojectionError=3.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < 10:
        return None
    selected = inliers.ravel()
    inlier_objects, inlier_images = objects[selected], images[selected]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(inlier_objects, inlier_images, target["K"], None, rvec, tvec)
    except cv2.error:
        pass
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    pnp_error = _feature_error(inlier_objects, inlier_images, T, target["K"])
    T, icp = _refine(T, source, target, inlier_objects, inlier_images)
    error = _feature_error(inlier_objects, inlier_images, T, target["K"])
    warped = _warp_mm(source["raw"], source["K"], target["K"], T, target["raw"].shape)
    overlap = (warped > 0) & (target["raw"] > 0)
    residual = np.abs(warped[overlap] - target["raw"][overlap])
    median_residual = float(np.median(residual)) if residual.size else math.inf
    overlap_count = int(overlap.sum())
    quality = len(inliers) * min(overlap_count / 100000.0, 1.0) / (1.0 + error) / (1.0 + median_residual / 10.0)
    return {"source_id": source.get("id", str(source.get("frame_id"))),
            "target_id": target.get("id", str(target.get("frame_id"))),
            "T_source_to_target": T, "transform_translation_unit": "mm",
            "detected_source_keypoints": len(source["keypoints"]),
            "detected_target_keypoints": len(target["keypoints"]),
            "ratio_or_mutual_matches": len(matches), "depth_valid_correspondences": len(objects),
            "pnp_inliers": len(inliers), "pnp_median_reprojection_px": pnp_error,
            "final_median_reprojection_px": error, "raw_depth_overlap_pixels": overlap_count,
            "raw_depth_median_residual_mm": median_residual, "quality_score": float(quality), **icp}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def select_reference(target: dict, candidate_ids: Iterable[int], load_frame: Callable[[int], dict]
                     ) -> tuple[dict | None, dict | None, list[dict]]:
    """Select one RGB-D reference, or return ``(None, None, attempts)``.

    Frames require true-RGB uint8 ``rgb``, metre ``raw_m`` and ``K``. Candidate
    failures are individually audited. The returned source is the original
    loader record with true RGB. Selection uses no optimized or GT depth.
    """
    target_geometry = _frame(target)
    attempts, usable = [], []
    target_id = target.get("frame_id")
    for source_id in dict.fromkeys(int(value) for value in candidate_ids):
        if source_id == target_id:
            attempts.append({"source_frame": source_id, "usable": False, "reason": "target_frame"})
            continue
        try:
            source = load_frame(source_id)
            source_geometry = _frame(source)
            if source_geometry["raw"].shape != target_geometry["raw"].shape:
                raise ValueError("Reference and target RGB-D resolutions differ")
            registration = _register(source_geometry, target_geometry)
        except (OSError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as error:
            attempts.append({"source_frame": source_id, "usable": False,
                             "reason": "candidate_error", "error": str(error)})
            continue
        if registration is None:
            attempts.append({"source_frame": source_id, "usable": False, "reason": "registration_failed"})
            continue
        T = registration["T_source_to_target"]
        translation = float(np.linalg.norm(T[:3, 3]))
        rotation = _rotation_degrees(T[:3, :3])
        benchmark = (registration["pnp_inliers"] >= 10
                     and registration["final_median_reprojection_px"] <= 5.0
                     and registration["raw_depth_overlap_pixels"] >= 10000
                     and registration["raw_depth_median_residual_mm"] <= 80.0)
        geometry = bool(np.isfinite(T).all() and 30.0 <= translation <= 350.0 and rotation <= 45.0)
        registration.update(source_frame=source_id, estimated_translation_mm=translation,
                            estimated_rotation_deg=rotation, benchmark_usable=bool(benchmark),
                            sequence_geometry_usable=geometry, usable=bool(benchmark and geometry),
                            automatic_view_score=float(registration["quality_score"] *
                                                       (0.5 + 0.5 * min(translation / 80.0, 1.0))))
        attempts.append(_json_value(registration))
        if benchmark and geometry:
            usable.append((registration, source))
    if not usable:
        return None, None, attempts
    registration, source = max(usable, key=lambda item: item[0]["automatic_view_score"])
    return registration, source, attempts


def refine_reference(target_depth_m, target_rgb, target_raw_m, source_depth_m,
                     source_rgb, K, T, *, source_K=None):
    """Continuous correction in metres; registration T translation is in mm.

    No accepted sensor values are restored. The frozen solver receives a copy
    of T converted to metres, leaving registration metadata and caller arrays intact.
    """
    from .continuous_multiview import refine
    transform = np.array(T, dtype=np.float64, copy=True)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-4)):
        raise ValueError('T must be a rigid source-to-target transform with mm translation')
    target_K = _intrinsics(K)
    reference_K = _intrinsics(source_K) if source_K is not None else target_K
    for rgb in (target_rgb, source_rgb):
        if np.asarray(rgb).dtype != np.uint8:
            raise ValueError('RGB must be uint8')
    transform[:3, 3] *= .001
    raw = np.asarray(target_raw_m, dtype=np.float32)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 0).astype(np.float32)
    reference = dict(depth_m=source_depth_m, rgb=source_rgb, K=reference_K, T_m=transform)
    return refine(target_depth_m, target_rgb, raw, [reference], target_K)
