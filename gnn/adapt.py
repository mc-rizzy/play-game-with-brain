"""Meta-Model Layer (few-shot adapter): personalize the GNN in seconds.

The base GNN trains once on pooled recordings, but every brain is wired a
little differently. During the 15-second Calibration Wizard each new player
produces a tiny labeled support set (~a few frames per command). This module
runs a MAML-style inner loop: freeze the spatial-GNN backbone, clone the
classification head, and take a few gradient steps on the support set, so
the game instantly understands *this* player's brain patterns.

Only the head (~2k params) moves, so adaptation finishes in milliseconds on
CPU and cannot destabilize the backbone. The adapted head + per-user gate
thresholds save to ``adapter.pt`` and load in ``gnn/live.py --adapter``.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from gnn.model import CHANNELS, CLASSES, EMA, CONFIDENCE, COOLDOWN_S


def default_thresholds():
    """Live-gate defaults (Muse personalizes these per player)."""
    return {"ema": EMA, "confidence": CONFIDENCE, "cooldown_s": COOLDOWN_S}


@torch.no_grad()
def _support_acc(model, X, y):
    logits = model(torch.tensor(np.asarray(X), dtype=torch.float32))
    pred = logits.argmax(1).numpy()
    return float((pred == np.asarray(y)).mean())


def _forward(model, X, A=None):
    if A is not None:
        return model(X, A_func=A)
    return model(X)


def adapt_head(model, X, y, A=None, steps=8, lr=0.05, l2=1e-4, seed=0):
    """Few-shot adapt a cloned model; returns (adapted, info).

    ``X``: (n, N, F) support features, ``y``: (n,) class ids, ``A``:
    optional (n, N, N) per-frame functional adjacency for SpatialGNN.
    Freezes the backbone and fine-tunes the head only (MAML-style
    inner loop).
    """
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("need a non-empty labeled support set")
    At = (torch.tensor(np.asarray(A, dtype=np.float32))
          if A is not None else None)
    if At is not None and len(At) != len(X):
        raise ValueError("A and X must have the same first dim")
    adapted = copy.deepcopy(model).cpu()
    adapted.eval()  # dropout/LN frozen: stable on tiny support sets
    for p in adapted.parameters():
        p.requires_grad_(False)
    for p in adapted.head.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        before_logits = _forward(
            adapted, torch.tensor(X), At).argmax(1).numpy()
    before = float((before_logits == y).mean())
    opt = torch.optim.SGD(adapted.head.parameters(), lr=lr,
                          weight_decay=l2)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    losses = []
    for _ in range(max(1, steps)):
        opt.zero_grad()
        loss = loss_fn(_forward(adapted, Xt, At), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    after = _support_acc(adapted, X, y) if At is None else float(
        (_forward(adapted, Xt, At).argmax(1).numpy() == y).mean())
    info = {"support_n": len(X), "classes": sorted(set(y.tolist())),
            "steps": steps, "lr": lr, "before_acc": before,
            "after_acc": after, "loss_first": losses[0],
            "loss_last": losses[-1], "dynamic_edges": At is not None}
    return adapted, info


def save_adapter(path, model, classes=None, channels=None, thresholds=None,
                 calibration=None, muse=None):
    """Save an adapted head + metadata -> adapter.pt dict (also returned)."""
    head = getattr(model, "head", None)
    if head is None:
        raise ValueError("model has no .head to save")
    adapter = {"head_state": {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()},
               "head_cls": type(head).__name__,
               "classes": list(classes or CLASSES),
               "channels": list(channels or getattr(model, "channels",
                                                    CHANNELS)),
               "fin": getattr(model, "gc1", None) and model.gc1.lin.in_features,
               "thresholds": dict(thresholds or default_thresholds()),
               "calibration": calibration or {},
               "muse": muse or {}}
    torch.save(adapter, path)
    return adapter


def load_adapter(path):
    """Load an adapter.pt dict saved by :func:`save_adapter`."""
    adapter = torch.load(path, map_location="cpu", weights_only=False)
    if "head_state" not in adapter:
        raise ValueError(f"{path}: not an adapter file (no head_state)")
    return adapter


def apply_adapter(model, adapter):
    """Load adapted head weights into a base model (in place, returned)."""
    model.head.load_state_dict(adapter["head_state"])
    return model
