"""Frozen balanced geometry rule; no learned confidence or annotation inputs."""
import torch
from typing import Tuple
@torch.no_grad()
def local_geometry_confidence(
    self,
    sparse_disparities: torch.Tensor,
    pred_disparities: torch.Tensor,
    sparse_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate prior points by local, affine-invariant geometry agreement.

    PDA aligns disparities with ``d_prior = scale * d_pred + shift``.  For a
    centre point i and a neighbouring prior point j, subtracting the centre
    cancels the unknown shift.  This method fits the remaining local scale
    between the two vectors of disparity differences and rejects centres
    whose robust residual or direction agreement is poor.

    The operation is deliberately performed only on sampled valid prior
    points and in chunks, so it does not build an H*W by N distance matrix.
    """
    radius = int(self.args.confidence_radius)
    max_neighbors = int(self.args.confidence_k)
    min_neighbors = int(self.args.confidence_min_neighbors)
    chunk_size = int(self.args.confidence_chunk_size)
    max_rel_error = float(self.args.confidence_max_rel_error)
    min_cosine = float(self.args.confidence_min_cosine)
    noise_floor_rel = float(self.args.confidence_noise_floor_rel)
    flat_rel = float(self.args.confidence_flat_rel)
    smooth_rel = float(self.args.confidence_smooth_rel)
    relaxed_max_error = float(self.args.confidence_relaxed_max_error)
    relaxed_min_cosine = float(self.args.confidence_relaxed_min_cosine)
    global_max_robust_z = float(self.args.confidence_global_max_robust_z)

    if radius < 1:
        raise ValueError("confidence_radius must be at least 1.")
    if not 2 <= min_neighbors <= max_neighbors:
        raise ValueError("Require 2 <= confidence_min_neighbors <= confidence_k.")
    if chunk_size < 1:
        raise ValueError("confidence_chunk_size must be positive.")

    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy != 0 or dx != 0
    ]
    offsets.sort(key=lambda item: (item[0] ** 2 + item[1] ** 2, item[0], item[1]))
    offset_y = torch.tensor([p[0] for p in offsets], device=self.device, dtype=torch.long)
    offset_x = torch.tensor([p[1] for p in offsets], device=self.device, dtype=torch.long)
    spatial_weight = torch.rsqrt(
        offset_y.float().square() + offset_x.float().square()
    )

    confidence = torch.zeros_like(sparse_disparities, dtype=torch.float32)
    relative_error_map = torch.full_like(sparse_disparities, torch.nan, dtype=torch.float32)
    cosine_map = torch.full_like(sparse_disparities, torch.nan, dtype=torch.float32)
    sensor_variation_map = torch.full_like(
        sparse_disparities, torch.nan, dtype=torch.float32
    )
    global_robust_z_map = torch.full_like(
        sparse_disparities, torch.nan, dtype=torch.float32
    )
    cosine_meaningful_map = torch.zeros_like(sparse_masks, dtype=torch.bool)
    decision_reason_map = torch.zeros_like(sparse_disparities, dtype=torch.uint8)
    evaluated = torch.zeros_like(sparse_masks, dtype=torch.bool)
    trusted = torch.zeros_like(sparse_masks, dtype=torch.bool)
    batch_size, height, width = sparse_masks.shape
    # Sparse depth is commonly stored in millimetres. Its disparity
    # differences can have squared energy around 1e-10, so float32 eps
    # (about 1e-7) is far too large for the degeneracy test here.
    eps = 1e-12
    stats = []

    for batch_id in range(batch_size):
        centres = torch.nonzero(sparse_masks[batch_id], as_tuple=False)
        all_sparse = sparse_disparities[batch_id][sparse_masks[batch_id]].float()
        all_pred = pred_disparities[batch_id][sparse_masks[batch_id]].float()
        design = torch.stack([all_pred, torch.ones_like(all_pred)], dim=1)
        fit_keep = torch.ones_like(all_sparse, dtype=torch.bool)
        global_scale = torch.tensor(1.0, device=self.device)
        global_shift = torch.tensor(0.0, device=self.device)
        # Robust whole-frame affine relation.  This catches a locally flat
        # but globally displaced flying patch without imposing a fixed
        # percentage of rejected points.
        for _ in range(3):
            solution = torch.linalg.lstsq(
                design[fit_keep], all_sparse[fit_keep, None]
            ).solution[:, 0]
            global_scale, global_shift = solution[0], solution[1]
            all_abs_residual = torch.abs(
                global_scale * all_pred + global_shift - all_sparse
            )
            residual_median = torch.median(all_abs_residual)
            residual_mad = torch.median(
                torch.abs(all_abs_residual - residual_median)
            )
            robust_sigma = (
                1.4826 * residual_mad +
                torch.median(all_sparse.abs()) * noise_floor_rel + eps
            )
            next_keep = all_abs_residual <= residual_median + 4.5 * robust_sigma
            if int(next_keep.sum().item()) < max(self.K + 1, 64):
                break
            fit_keep = next_keep
        all_abs_residual = torch.abs(
            global_scale * all_pred + global_shift - all_sparse
        )
        residual_median = torch.median(all_abs_residual)
        residual_mad = torch.median(torch.abs(all_abs_residual - residual_median))
        robust_sigma = (
            1.4826 * residual_mad +
            torch.median(all_sparse.abs()) * noise_floor_rel + eps
        )
        all_global_robust_z = torch.clamp_min(
            all_abs_residual - residual_median, 0.0
        ) / robust_sigma
        global_robust_z_map[
            batch_id, centres[:, 0], centres[:, 1]
        ] = all_global_robust_z

        for start in range(0, centres.shape[0], chunk_size):
            chunk = centres[start:start + chunk_size]
            centre_y, centre_x = chunk[:, 0], chunk[:, 1]
            global_robust_z = all_global_robust_z[start:start + chunk.shape[0]]

            raw_y = centre_y[:, None] + offset_y[None, :]
            raw_x = centre_x[:, None] + offset_x[None, :]
            in_bounds = (
                (raw_y >= 0) & (raw_y < height) &
                (raw_x >= 0) & (raw_x < width)
            )
            near_y = raw_y.clamp(0, height - 1)
            near_x = raw_x.clamp(0, width - 1)
            valid = in_bounds & sparse_masks[batch_id, near_y, near_x]

            # Offsets are distance-sorted. Keep the nearest K valid points.
            selected = valid & (valid.cumsum(dim=1) <= max_neighbors)
            neighbor_count = selected.sum(dim=1)

            sparse_center = sparse_disparities[
                batch_id, centre_y, centre_x
            ][:, None]
            pred_center = pred_disparities[
                batch_id, centre_y, centre_x
            ][:, None]
            sparse_delta = (
                sparse_disparities[batch_id, near_y, near_x] - sparse_center
            )
            pred_delta = (
                pred_disparities[batch_id, near_y, near_x] - pred_center
            )

            weights = selected.float() * spatial_weight[None, :]
            cross = (weights * pred_delta * sparse_delta).sum(dim=1)
            pred_energy = (weights * pred_delta.square()).sum(dim=1)
            sparse_energy = (weights * sparse_delta.square()).sum(dim=1)
            local_scale = cross / (pred_energy + eps)

            abs_residual = torch.abs(sparse_delta - local_scale[:, None] * pred_delta)
            abs_residual = abs_residual.masked_fill(~selected, torch.nan)
            abs_signal = torch.abs(sparse_delta).masked_fill(~selected, torch.nan)
            abs_pred_signal = torch.abs(pred_delta).masked_fill(~selected, torch.nan)
            robust_residual = torch.nanmedian(abs_residual, dim=1).values
            robust_signal = torch.nanmedian(abs_signal, dim=1).values
            robust_pred_signal = torch.nanmedian(abs_pred_signal, dim=1).values

            # V6.2 divided only by ``robust_signal``.  On a valid planar
            # surface, quantised RGB-D depth often makes this value exactly
            # zero; a sub-millimetre residual was then divided by 1e-12 and
            # a normal point looked catastrophically inconsistent.  V6.3
            # uses a scale-invariant floor tied to the centre disparity.
            sparse_c = sparse_center.squeeze(1).abs()
            pred_c = pred_center.squeeze(1).abs()
            sensor_floor = (sparse_c * noise_floor_rel).clamp_min(eps)
            pred_floor = (pred_c * noise_floor_rel).clamp_min(eps)
            relative_error = robust_residual / torch.maximum(
                robust_signal, sensor_floor
            )
            sensor_variation = robust_signal / (sparse_c + eps)
            cosine = cross / torch.sqrt(pred_energy * sparse_energy + eps)

            can_evaluate = (
                (neighbor_count >= min_neighbors) &
                torch.isfinite(relative_error) &
                torch.isfinite(sensor_variation)
            )

            # The signed cosine is meaningful only when both local signals
            # exceed the sensor/model numerical floor.  Flat areas are
            # judged by sensor self-consistency instead of an unstable
            # angle between two nearly zero vectors.
            cosine_meaningful = (
                (robust_signal > sensor_floor) &
                (robust_pred_signal > pred_floor) &
                torch.isfinite(cosine)
            )
            flat_sensor = sensor_variation <= flat_rel
            global_consistent = global_robust_z <= global_max_robust_z
            strict_geometry = (
                can_evaluate &
                (local_scale > 0.) &
                (relative_error <= max_rel_error) &
                cosine_meaningful &
                (cosine >= min_cosine)
            )
            smooth_geometry = (
                can_evaluate &
                (sensor_variation <= smooth_rel) &
                global_consistent &
                (local_scale > 0.) &
                (relative_error <= relaxed_max_error) &
                cosine_meaningful &
                (cosine >= relaxed_min_cosine)
            )
            flat_rescue = can_evaluate & flat_sensor & global_consistent
            chunk_trusted = flat_rescue | strict_geometry | smooth_geometry

            strict_error_score = torch.exp(
                -torch.square(relative_error / max(max_rel_error, 1e-6))
            )
            strict_cosine_score = torch.where(
                cosine_meaningful,
                ((cosine - min_cosine) / max(1. - min_cosine, 1e-6)).clamp(0., 1.),
                torch.zeros_like(cosine),
            )
            geometry_score = strict_error_score * strict_cosine_score
            flat_score = torch.exp(-torch.square(sensor_variation / max(flat_rel, 1e-6)))
            smooth_score = (
                torch.exp(-torch.square(relative_error / max(relaxed_max_error, 1e-6))) *
                torch.where(
                    cosine_meaningful,
                    ((cosine - relaxed_min_cosine) /
                     max(1. - relaxed_min_cosine, 1e-6)).clamp(0., 1.),
                    torch.zeros_like(cosine),
                )
            )
            chunk_confidence = torch.maximum(
                geometry_score, torch.maximum(flat_score, smooth_score)
            ).clamp(0., 1.)

            reason = torch.zeros_like(neighbor_count, dtype=torch.uint8)
            reason = torch.where(flat_rescue, torch.ones_like(reason), reason)
            reason = torch.where(
                strict_geometry & ~flat_rescue,
                torch.full_like(reason, 2),
                reason,
            )
            reason = torch.where(
                smooth_geometry & ~flat_rescue & ~strict_geometry,
                torch.full_like(reason, 3),
                reason,
            )

            confidence[batch_id, centre_y, centre_x] = torch.where(
                can_evaluate, chunk_confidence, torch.zeros_like(chunk_confidence)
            )
            relative_error_map[batch_id, centre_y, centre_x] = torch.where(
                can_evaluate, relative_error, torch.full_like(relative_error, torch.nan)
            )
            cosine_map[batch_id, centre_y, centre_x] = torch.where(
                can_evaluate, cosine, torch.full_like(cosine, torch.nan)
            )
            sensor_variation_map[batch_id, centre_y, centre_x] = torch.where(
                can_evaluate,
                sensor_variation,
                torch.full_like(sensor_variation, torch.nan),
            )
            cosine_meaningful_map[batch_id, centre_y, centre_x] = (
                can_evaluate & cosine_meaningful
            )
            decision_reason_map[batch_id, centre_y, centre_x] = reason
            evaluated[batch_id, centre_y, centre_x] = can_evaluate
            trusted[batch_id, centre_y, centre_x] = chunk_trusted

        total_count = int(sparse_masks[batch_id].sum().item())
        evaluated_count = int(evaluated[batch_id].sum().item())
        trusted_count = int(trusted[batch_id].sum().item())
        reason_values = decision_reason_map[batch_id]
        flat_count = int((reason_values == 1).sum().item())
        strict_count = int((reason_values == 2).sum().item())
        smooth_count = int((reason_values == 3).sum().item())
        eval_errors = relative_error_map[batch_id][evaluated[batch_id]]
        eval_cosines = cosine_map[batch_id][evaluated[batch_id]]
        quantile_levels = torch.tensor([0.25, 0.5, 0.75], device=self.device)
        if eval_errors.numel() == 0:
            error_quantiles = [float("nan")] * 3
            cosine_quantiles = [float("nan")] * 3
        else:
            error_quantiles = torch.quantile(eval_errors, quantile_levels).tolist()
            cosine_quantiles = torch.quantile(eval_cosines, quantile_levels).tolist()
        required_count = min(int(self.args.confidence_min_keep), total_count)
        if trusted_count < max(required_count, self.K + 1):
            raise RuntimeError(
                "Local confidence filtering retained too few prior points: "
                f"batch={batch_id}, total={total_count}, evaluated={evaluated_count}, "
                f"trusted={trusted_count}, error_q25/50/75={error_quantiles}, "
                f"cosine_q25/50/75={cosine_quantiles}. Increase pattern/radius, "
                "or relax confidence_max_rel_error/confidence_min_cosine."
            )
        stats.append({
            "batch": batch_id,
            "total": total_count,
            "evaluated": evaluated_count,
            "trusted": trusted_count,
            "trusted_ratio": trusted_count / max(total_count, 1),
            "flat_rescue": flat_count,
            "strict_geometry": strict_count,
            "smooth_rescue": smooth_count,
            "global_affine_scale": float(global_scale.item()),
            "global_affine_shift": float(global_shift.item()),
            "global_affine_inliers": int(fit_keep.sum().item()),
            "error_quantiles": error_quantiles,
            "cosine_quantiles": cosine_quantiles,
        })

    self.last_confidence_map = confidence.detach().cpu()
    self.last_confident_mask = trusted.detach().cpu()
    self.last_confidence_evaluated_mask = evaluated.detach().cpu()
    self.last_input_sparse_mask = sparse_masks.detach().cpu()
    self.last_confidence_relative_error = relative_error_map.detach().cpu()
    self.last_confidence_cosine = cosine_map.detach().cpu()
    self.last_confidence_sensor_variation = sensor_variation_map.detach().cpu()
    self.last_confidence_global_robust_z = global_robust_z_map.detach().cpu()
    self.last_confidence_cosine_meaningful = cosine_meaningful_map.detach().cpu()
    self.last_confidence_decision_reason = decision_reason_map.detach().cpu()
    self.last_confidence_stats = stats
    return confidence, trusted, evaluated
