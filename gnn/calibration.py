"""Calibration orchestration: 15 s of cues -> adapted head + Muse coaching.

The Calibration Wizard (``game/calibrate.py``) shows each cue while this
module collects labeled GNN frames through a caller-provided ``read_frame``
callback, runs the MAML-style head adaptation (``gnn/adapt.py``), asks Muse
(``gnn/muse.py``) to coach the player and tune the live gates, and saves an
``adapter.pt`` + ``calibration_report.json`` pair.

``read_frame`` abstracts the EEG source, so the same code calibrates from
live hardware, replayed recordings, or synthetic demo data::

    frame = read_frame()  # -> (features (N, F), epoch (C, T) | None)
                          #    or None when no frame is ready yet
"""

import json
import os
import time

import numpy as np

from gnn.model import CLASSES

# 5 cues x 3 s = 15 s total. Order alternates hemispheres so LEFT/RIGHT
# motor imagery (C3/C4 mu/beta ERD) separates cleanly.
CUES_15S = (("LEFT", 3.0), ("RIGHT", 3.0), ("UP", 3.0), ("DOWN", 3.0),
            ("INTERACT", 3.0))

LABEL_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}


def silhouette_separability(X, y):
    """Silhouette-like class separation in [0, 1] for the Muse prompt."""
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    if len(classes) < 2 or len(X) < 2:
        return 0.0
    centroids = {c: X[y == c].mean(axis=0) for c in classes}
    scores = []
    for vec, lab in zip(X, y):
        d_own = float(np.linalg.norm(vec - centroids[int(lab)]))
        others = [float(np.linalg.norm(vec - centroids[c]))
                  for c in classes if c != int(lab)]
        d_other = min(others)
        denom = d_other + d_own + 1e-9
        scores.append(max(0.0, (d_other - d_own) / denom))
    return float(np.mean(scores))


def summarize_support(X, y, classes=None):
    """Compact calibration summary for the Muse prompt + report."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    classes = list(classes or CLASSES)
    per_class = {}
    for c in sorted(set(y.tolist())):
        name = classes[c] if c < len(classes) else f"class_{c}"
        per_class[name] = {"n": int((y == c).sum()),
                           "mean": [round(float(v), 3)
                                    for v in X[y == c].mean(axis=0).mean(
                                        axis=0).tolist()]}
    return {"n_frames": len(X), "n_classes": len(per_class),
            "feature_shape": list(X.shape[1:]),
            "separability": round(silhouette_separability(X, y), 3),
            "per_class": per_class}


def collect_cue_frames(read_frame, cue, seconds, poll_s=0.25, time_fn=None,
                       sleep_fn=None):
    """Poll ``read_frame`` for ``seconds``; returns (frames, labels, epochs)."""
    time_fn = time_fn or time.time
    sleep_fn = sleep_fn or time.sleep
    label = LABEL_TO_INDEX[cue]
    frames, labels, epochs = [], [], []
    deadline = time_fn() + seconds
    while time_fn() < deadline:
        frame = read_frame()
        if frame is not None:
            feats, epoch = frame
            frames.append(np.asarray(feats, dtype=np.float32))
            labels.append(label)
            epochs.append(epoch)
        sleep_fn(poll_s)
    return frames, labels, epochs


def run_calibration(read_frame, model, cues=CUES_15S, adapt_steps=8,
                    adapt_lr=0.05, muse_client=None, out_dir=None,
                    on_cue=None, poll_s=0.25):
    """Full wizard backend: collect -> adapt -> Muse -> save.

    ``on_cue(cue, seconds, index, total)`` fires as each cue starts (the
    Pygame wizard uses it to update the screen). Returns a result dict.
    """
    from gnn import adapt as adapt_mod
    from gnn.muse import MuseClient

    cues = list(cues)
    X, y, E = [], [], []
    for i, (cue, seconds) in enumerate(cues):
        if on_cue is not None:
            on_cue(cue, seconds, i, len(cues))
        hook = getattr(read_frame, "set_cue", None)
        if hook is not None:
            hook(cue)
        frames, labels, epochs = collect_cue_frames(read_frame, cue, seconds,
                                                    poll_s=poll_s)
        X.extend(frames)
        y.extend(labels)
        E.extend(epochs)
    if len(X) < len(cues):
        raise ValueError(f"only {len(X)} frames collected; "
                         f"need >= {len(cues)} (one per cue)")
    Xa = np.stack(X).astype(np.float32)
    ya = np.array(y, dtype=np.int64)
    A = None
    if all(e is not None for e in E):
        from gnn.graph import functional_adj
        A = np.stack([functional_adj(e) for e in E]).astype(np.float32)
    adapted, adapt_info = adapt_mod.adapt_head(model, Xa, ya, A=A,
                                               steps=adapt_steps,
                                               lr=adapt_lr)
    summary = summarize_support(Xa, ya, getattr(model, "classes", CLASSES)
                                if hasattr(model, "classes") else CLASSES)
    summary["adapt"] = {k: (round(float(v), 4)
                            if isinstance(v, float) else v)
                        for k, v in adapt_info.items()
                        if k in ("before_acc", "after_acc", "steps",
                                 "support_n")}
    muse = muse_client or MuseClient()
    muse_result = muse.analyze_calibration(summary)
    result = {"summary": summary, "adapt_info": adapt_info,
              "muse": muse_result, "muse_calls": list(muse.calls),
              "thresholds": muse_result["thresholds"]}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        adapter_path = os.path.join(out_dir, "adapter.pt")
        report_path = os.path.join(out_dir, "calibration_report.json")
        adapt_mod.save_adapter(
            adapter_path, adapted,
            classes=getattr(model, "classes", CLASSES)
            if hasattr(model, "classes") else CLASSES,
            channels=getattr(model, "channels", None),
            thresholds=muse_result["thresholds"],
            calibration={"adapt": adapt_info, "summary": summary},
            muse={"result": muse_result, "calls": list(muse.calls)})
        with open(report_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        result["adapter_path"] = adapter_path
        result["report_path"] = report_path
    else:
        result["adapter_path"] = None
        result["report_path"] = None
    result["adapted_model"] = adapted
    return result


def synthetic_read_frame(n_ch=8, fin=12, seed=0):
    """Demo-mode frame source: class-separable fake features, no hardware.

    Returns a stateful ``read_frame`` whose distribution shifts when
    ``set_cue(cue)`` is called — the wizard calls it per cue.
    """
    rng = np.random.RandomState(seed)
    state = {"cue": "LEFT"}
    shifts = {c: rng.randn(n_ch, fin).astype(np.float32) * 0.4
              for c in CLASSES}

    def read_frame():
        base = shifts[state["cue"]]
        feats = base + rng.randn(n_ch, fin).astype(np.float32) * 0.5
        return feats, None

    def set_cue(cue):
        state["cue"] = cue

    read_frame.set_cue = set_cue
    return read_frame
