"""Train the EEG command classifier on EEG_recordings/.

Run after collecting more data (from the workspace root, Windows)::

    .venv/Scripts/python -m gnn.train --data EEG_recordings --out gnn/checkpoints

Needs at least two commands with trials each; ~30 trials/command is the
calibration target (see EEG_recordings/README.md). Splits whole trials
80/20 stratified — windows from one trial never land on both sides.
Saves ``model.pt`` (weights + channels/classes) and ``config.json``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gnn import recordings
from gnn.dataset import build_dataset, trial_split, CLASSES
from gnn.model import (CHANNELS, FS, N_FEATURES, EEGGraphNet,
                       extract_node_features, train)


def featurize(X, mains=50):
    feats = [extract_node_features(ep, fs=FS, mains=mains) for ep in X]
    return np.stack(feats).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train EEG command GNN")
    ap.add_argument("--data", default="EEG_recordings")
    ap.add_argument("--out", default="gnn/checkpoints")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--mains", type=int, default=50,
                    help="mains notch frequency (50 EU / 60 US)")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--step-s", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    trials = recordings.load_labeled(args.data, expected_channels=CHANNELS,
                                     expected_fs=FS)
    labels = sorted({t["label"] for t in trials})
    print(f"Loaded {len(trials)} trials, labels: {labels}")
    if len(labels) < 2:
        raise SystemExit("Need trials from at least 2 commands to train.")
    counts = {lab: sum(1 for t in trials if t["label"] == lab)
              for lab in labels}
    print(f"Trials per command: {counts}")

    X, y, tids = build_dataset(trials, args.window_s, args.step_s, FS)
    print(f"Windows: {len(X)} x {X.shape[1:]}")
    tr, va = trial_split(tids, y, args.val_fraction, args.seed)
    print(f"Train windows: {len(tr)}, val windows: {len(va)} "
          f"({len(set(tids[tr]))}/{len(set(tids[va]))} trials)")

    print("Extracting features...")
    F = featurize(X, args.mains)
    print(f"Features: {F.shape}")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = EEGGraphNet(channels=CHANNELS, fin=N_FEATURES,
                        hidden=args.hidden,
                        n_classes=len(CLASSES)).to(device)
    os.makedirs(args.out, exist_ok=True)
    history, best = train(model, F[tr], y[tr], F[va], y[va],
                          epochs=args.epochs, batch=args.batch, lr=args.lr,
                          seed=args.seed,
                          save_path=os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump({"channels": CHANNELS, "classes": CLASSES, "fs": FS,
                   "fin": N_FEATURES, "hidden": args.hidden,
                   "mains": args.mains, "window_s": args.window_s,
                   "step_s": args.step_s, "best_val_acc": best,
                   "n_trials": len(trials), "labels": labels}, fh, indent=2)
    print(f"Best val acc: {best:.3f} -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
