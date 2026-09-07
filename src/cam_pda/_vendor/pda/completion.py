# Extracted from the evaluated PDA completion implementation; see notices.
# Modified for explicit weights/device configuration and Python/Torch KNN.
import torch
import warnings
from contextlib import nullcontext
from typing import Dict, Tuple, Optional
from types import SimpleNamespace
from .utils import depth2disparity, disparity2depth
torch_cluster = None

class DepthCompletion(torch.nn.Module):
    def __init__(self, mde_path, device):
        super().__init__()
        self.device = torch.device(device)
        self.K = 5
        self.args = SimpleNamespace(frozen_model_size='vitb', K=5, confidence_filter=False)
        self.depth_model = self.init_depth_model(mde_path)

    def init_depth_model(self, fmde_path):
        """ We implement @depth-anything-v2 here, you can replace it with other depth estimation models. (like VGGT or moge ...)"""
        from .depth_anything_v2 import build_backbone
        depth_model = build_backbone(
            depth_size=self.args.frozen_model_size
        )
        state_dict = torch.load(fmde_path, map_location='cpu', weights_only=True)
        depth_model.load_state_dict(state_dict=state_dict)

        depth_model.construct_aux_layers()
        depth_model.freeze_network({'encoder', 'decoder'})
        depth_model = depth_model.eval().to(self.device)

        return depth_model


    def calc_scale_shift(self,
        k_sparse_targets: torch.Tensor,
        k_pred_targets: torch.Tensor,
        currk_dists: Optional[torch.Tensor] = None,
        knn: bool = False
    ):
        k_pred_targets += torch.rand(*k_pred_targets.shape, device=self.device) * 1e-5
        X = torch.stack([k_pred_targets, torch.ones_like(k_pred_targets, device=self.device)], dim=2)

        # To perform weights to the knn points.
        if knn > 0: k_sparse_targets, X = self.perform_weighted(k_sparse_targets, X, currk_dists)
        elif k_pred_targets.shape[0] > 1: k_sparse_targets = k_sparse_targets.unsqueeze(-1)

        solution = torch.linalg.lstsq(X, k_sparse_targets)
        scale, shift = solution[0][:, 0].squeeze(), solution[0][:, 1].squeeze()

        return scale, shift

    def perform_weighted(self,
        sparse_ori : torch.Tensor,
        pred_ori : torch.Tensor,
        dists : torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform weighted operations on input tensors using distance-based weights. A diagonal
        matrix is created from the normalized weights and used to weight the inputs.

        Notes:
            - Weights are calculated as the inverse of the distances.
            - Weights are normalized to ensure they sum to 1.

        Args:
            sparse_ori (torch.Tensor): Sparse original map.
            pred_ori (torch.Tensor): Predicted map.
            dists (torch.Tensor): Distances used for weight calculation.

        Returns:
            Tuple: Containing two tensors:
                - sparse_weighted: The weighted version of the sparse original map.
                - pred_weighted: The weighted version of the predicted map.
        """

        weights = 1 / dists
        wsum = weights.sum(dim=1, keepdim=True)
        weights = weights / wsum
        W = torch.diag_embed(weights)

        pred_weighted = W @ pred_ori
        sparse_weighted = W @ sparse_ori.unsqueeze(-1)
        return sparse_weighted, pred_weighted

    def knn_aligns(self,
        sparse_disparities: torch.Tensor,
        pred_disparities: torch.Tensor,
        sparse_masks: torch.Tensor,
        complete_masks: torch.Tensor,
        K: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform K-Nearest Neighbors (KNN) alignment on sparse and predicted disparities.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            complete_masks (torch.Tensor): Indicating which points in the map to be completed.
            K (int): The number of nearest neighbors to find for each map point.

        Returns:
            Tuple: Containing three tensors:
                - dists: The Euclidean distances from each sparse point to its K nearest neighbors.
                - k_sparse_targets: Disparities of the K nearest neighbors from the sparse data.
                - k_pred_targets: Disparities of the K nearest neighbors from the predicted data.
        """

        # Coordinates are processed to ensure compatibility with the KNN function.
        batch_sparse = torch.nonzero(sparse_masks, as_tuple=False)[..., [0, 2, 1]].float() # [N, 3] (b, x, y)
        batch_complete = torch.nonzero(complete_masks, as_tuple=False)[..., [0, 2, 1]].float() # [M, 3] (b, x, y)

        batch_x, batch_y = batch_sparse[:, 0].contiguous(), batch_complete[:, 0].contiguous()
        x, y = batch_sparse[:, -2:].contiguous(), batch_complete[:, -2:].contiguous()

        # Use `torch_cluster.knn` to find K nearest neighbors.
        with torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext():
            if torch_cluster is None:
                knn_map = self._torch_knn(x, y, batch_x, batch_y, K)
            else:
                knn_map = torch_cluster.knn(x=x, y=y, k=K, batch_x=batch_x, batch_y=batch_y) # [2, M * K]
        knn_indices = knn_map[1, :].view(-1, K)

        k_sparse_targets = sparse_disparities[sparse_masks][knn_indices]
        k_pred_targets = pred_disparities[sparse_masks][knn_indices]

        knn_coords = x[knn_indices]
        expanded_complete_points = y.unsqueeze(dim=1).repeat(1, K, 1)
        dists = torch.norm(expanded_complete_points - knn_coords, dim=2)

        return dists, k_sparse_targets, k_pred_targets

    def kss_completer(self,
        sparse_disparities: torch.Tensor,
        pred_disparities: torch.Tensor,
        complete_masks: torch.Tensor,
        sparse_masks: torch.Tensor,
        K: int = 5
    ) -> torch.Tensor:
        """
        Perform K-Nearest Neighbors (KNN) interpolation to complete sparse disparities.Use a batch-oriented
        implementation of KNN interpolation to complete the sparse disparities. We leverages "torch_cluster.knn"
        for acceleration and GPU memory efficiency.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map.
            pred_disparities (torch.Tensor): Dredicted disparities for sparse map points.
            complete_masks (torch.Tensor): Indicating which points in the complete map are valid.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            K (int): The number of nearest neighbors to use for interpolation. Defaults to 5.

        Returns:
            The completed disparities, interpolated from the nearest neighbors.
        """

        if not bool(complete_masks.any()):
            return torch.where(sparse_masks, sparse_disparities, torch.zeros_like(sparse_disparities)).float()

        # Use `knn_aligns` to find the K nearest neighbors and calculate distances.
        bottomk_dists, k_sparse_targets, k_pred_targets = self.knn_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks,
            complete_masks=complete_masks,
            K=K
        )

        scaled_preds = torch.zeros_like(sparse_disparities, device=self.device, dtype=torch.float32)
        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets,
            k_pred_targets=k_pred_targets,
            currk_dists=bottomk_dists,
            knn=True
        )

        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds[complete_masks] = pred_disparities[complete_masks] * scale + shift
        # The completed disparities are computed by combining the scaled predictions and the original sparse disparities.
        scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]
        return scaled_preds

    def global_aligns(self,
        sparse_disparities: torch.Tensor,
        pred_disparities: torch.Tensor,
        sparse_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform global alignment on sparse and predicted disparities. Extract the valid disparities from
        both sparse and predicted map based on the sparse masks.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.

        Returns:
            Tuple[torch.Tensor]: Containing two tensors:
                - k_sparse_targets: The valid disparities from the sparse map.
                - k_pred_targets: The valid disparities from the predicted map.
        """

        # The valid disparities are extracted and unsqueezed to maintain consistent dimensions.
        k_sparse_targets = sparse_disparities[sparse_masks].unsqueeze(dim=0)
        k_pred_targets = pred_disparities[sparse_masks].unsqueeze(dim=0)

        return k_sparse_targets, k_pred_targets

    def ss_completer(self,
        sparse_disparities: torch.Tensor,
        pred_disparities: torch.Tensor,
        sparse_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Complete sparse disparities using a simple scaling and shifting approach. Perform a global
        alignment of the sparse and predicted disparities, then applies a scaling and shifting
        transformation to complete the sparse disparities.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.

        Returns:
            The completed disparities, computed by scaling and shifting the predicted disparities.
        """

        # Use `global_aligns` to extract valid disparities.
        k_sparse_targets, k_pred_targets = self.global_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks
        )

        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets,
            k_pred_targets=k_pred_targets
        )

        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds = pred_disparities * scale + shift
        return scaled_preds

    @staticmethod
    def _torch_knn(x,y,batch_x,batch_y,k,chunk_size=8192):
        """Keep the original operations/chunks; free each distance matrix before the next."""
        rows,cols=[],[]
        for batch_id in torch.unique(batch_y):
            x_idx=torch.nonzero(batch_x==batch_id,as_tuple=False).squeeze(1)
            y_idx=torch.nonzero(batch_y==batch_id,as_tuple=False).squeeze(1)
            if x_idx.numel()<k:raise ValueError(f'Need at least {k} valid prior-depth points.')
            for start in range(0,y_idx.numel(),chunk_size):
                query_idx=y_idx[start:start+chunk_size]
                distances=torch.cdist(y[query_idx],x[x_idx])
                nearest=distances.topk(k,dim=1,largest=False).indices
                del distances
                rows.append(query_idx.repeat_interleave(k));cols.append(x_idx[nearest.reshape(-1)])
        return torch.stack([torch.cat(rows),torch.cat(cols)],dim=0)
