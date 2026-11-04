"""Spatial GNN Layer: electrodes as graph nodes, connectivity as edges.

Nodes = EEG electrodes in cap order (``gnn.model.CHANNELS``), each carrying
the Feature Extraction Layer's PSD node features. Edges mix two signals:

- spatial proximity: Gaussian kernel over 10-20 cap coordinates (static,
  always available, encodes the physical layout);
- functional connectivity: absolute Pearson correlation between the
  channels' recent time series (dynamic, per-frame, captures which brain
  regions fire together during *this* thought).

The GNN message-passes over the blended adjacency and pools to a
spatial-temporal latent vector; the (meta-adapted) head maps it to commands.
"""

import numpy as np
import torch
import torch.nn.functional as F

from gnn.model import CHANNELS, COORDS_2D, EEGGraphNet


def spatial_adj(channels=None, sigma=0.7):
    """Static adjacency from cap geometry: (N, N) float32 in [0, 1]."""
    channels = list(channels or CHANNELS)
    pts = np.array([COORDS_2D[c] for c in channels])
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    A = np.exp(-d ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)


def functional_adj(epoch, threshold=0.15):
    """Dynamic adjacency from functional connectivity: (N, N) float32.

    ``epoch`` is (C, T) filtered time series; edges are absolute Pearson
    correlations, thresholded for sparsity, with unit diagonal.
    """
    epoch = np.asarray(epoch, dtype=np.float64)
    corr = np.corrcoef(epoch)
    corr = np.nan_to_num(corr, nan=0.0)
    A = np.abs(corr).astype(np.float32)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 1.0)
    return A


def combined_adj(A_spatial, A_func=None, alpha=0.5):
    """Blend static + dynamic adjacency: (1-a)*spatial + a*functional."""
    A = np.asarray(A_spatial, dtype=np.float32)
    if A_func is None:
        return A
    B = np.asarray(A_func, dtype=np.float32)
    if B.shape != A.shape:
        raise ValueError(f"adjacency shape mismatch {B.shape} vs {A.shape}")
    C = (1.0 - alpha) * A + alpha * B
    np.fill_diagonal(C, 1.0)
    return C.astype(np.float32)


class SpatialGNN(EEGGraphNet):
    """EEGGraphNet with per-frame functional-connectivity edges.

    ``forward(x)`` behaves exactly like the parent (static + learned
    edges). Pass ``A_func`` (the current frame's functional adjacency)
    to route messages through electrodes that fire together *now*.
    :meth:`encode` returns the pre-head spatial-temporal latent vector.
    """

    def __init__(self, *args, func_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.func_weight = func_weight

    def _mix(self, A_func, batch, device):
        A_dyn = F.relu(torch.tanh(self.E1 @ self.E2.T - self.E2 @ self.E1.T))
        A = self.A_static + 0.5 * A_dyn  # (N, N)
        if A_func is None:
            return A.unsqueeze(0).expand(batch, -1, -1)
        if not torch.is_tensor(A_func):
            A_func = torch.tensor(np.asarray(A_func, dtype=np.float32))
        A_func = A_func.to(device)
        if A_func.dim() == 2:  # one shared frame
            A = A + self.func_weight * A_func
            return A.unsqueeze(0).expand(batch, -1, -1)
        # per-sample frames (B, N, N)
        return A.unsqueeze(0).expand(batch, -1, -1) + self.func_weight * A_func

    def encode(self, x, A_func=None):
        """(B, N, F) -> (B, hidden*2) spatial-temporal latent vector."""
        A = self._mix(A_func, x.size(0), x.device)
        h1 = self.drop(self.gc1(x, A))
        h2 = self.gc2(h1, A) + h1  # residual
        return torch.cat([h2.mean(1), h2.max(1)[0]], dim=-1)

    def forward(self, x, A_func=None):  # (B, N, F) -> (B, n_classes)
        return self.head(self.encode(x, A_func))


def as_spatial(model, func_weight=0.5):
    """Upgrade an EEGGraphNet to SpatialGNN, keeping trained weights.

    Architectures match (same backbone + head), so the state dict loads
    verbatim; only the forward gains per-frame functional edges.
    """
    if isinstance(model, SpatialGNN):
        model.func_weight = func_weight
        return model
    hidden = model.gc1.lin.out_features
    fin = model.gc1.lin.in_features
    n_classes = model.head[-1].out_features
    spatial = SpatialGNN(channels=list(getattr(model, "channels", CHANNELS)),
                         fin=fin, hidden=hidden, n_classes=n_classes,
                         func_weight=func_weight)
    spatial.load_state_dict(model.state_dict())
    spatial.eval()
    return spatial


def frame_graph(frame_features, epoch=None, channels=None,
                threshold=0.15, alpha=0.5):
    """Bundle one GNN frame: (features (N, F), adjacency (N, N)).

    ``frame_features`` comes from the Feature Extraction Layer;
    ``epoch`` (optional (C, T) filtered series) adds functional edges.
    """
    feats = np.asarray(frame_features, dtype=np.float32)
    A_sp = spatial_adj(channels)
    if feats.shape[0] != A_sp.shape[0]:
        raise ValueError(f"{feats.shape[0]} feature rows vs "
                         f"{A_sp.shape[0]} graph nodes")
    A_fn = functional_adj(epoch, threshold) if epoch is not None else None
    return feats, combined_adj(A_sp, A_fn, alpha)
