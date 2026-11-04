"""Train the EEG command GNN on Modal (cloud) instead of locally.

Same data, same model, same ``model.pt`` output as ``gnn/train.py`` — only
the number-crunching moves to a Modal worker. Recording loading and feature
extraction run locally (fast, tiny payloads); the GNN fit runs remotely.

Install once: .venv/Scripts/pip install modal

Auth (pick one)::

    A. .venv/Scripts/modal setup     # recommended: stores login, no keys to manage
    B. .env file with both values (read automatically on `modal run`):
         MODAL_TOKEN_ID=ak-...
         MODAL_TOKEN_SECRET=as-...
       Get both at modal.com -> Settings -> API tokens. A lone MODAL_TOKEN
       is only the ID and is not enough by itself.

Run (from the workspace root, Windows)::

    .venv/Scripts/modal run gnn/modal_train.py --data EEG_recordings --out gnn/checkpoints

Options mirror ``gnn/train.py`` (``--epochs --batch --lr --mains
--window-s --step-s --hidden --seed``). To use a GPU worker instead of CPU::

    set MODAL_GPU=T4 && .venv/Scripts/modal run gnn/modal_train.py ...

Out of Modal credits? Run the identical training on your own machine::

    .venv/Scripts/python -m gnn.train --data EEG_recordings --out gnn/checkpoints
"""

import os

try:
    import modal
except ImportError as exc:
    raise SystemExit(
        "The modal package is not installed. Run: pip install modal\n"
        "Or train locally instead: python -m gnn.train "
        "--data EEG_recordings --out gnn/checkpoints") from exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(root=ROOT):
    """Minimal KEY=VALUE .env parser (no dependency). Returns dict."""
    out = {}
    try:
        with open(os.path.join(root, ".env")) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def ensure_modal_auth(root=ROOT, strict=True):
    """Export Modal credentials from .env (existing env vars always win).

    Reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; a lone MODAL_TOKEN is
    accepted as the token ID. With strict=True, an ID without a secret
    (and no `modal setup` login) fails fast with instructions instead of
    a cryptic auth error from the API call.
    """
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return "env"
    file_vars = load_dotenv(root)
    token_id = (os.environ.get("MODAL_TOKEN_ID")
                or file_vars.get("MODAL_TOKEN_ID")
                or file_vars.get("MODAL_TOKEN"))
    secret = (os.environ.get("MODAL_TOKEN_SECRET")
              or file_vars.get("MODAL_TOKEN_SECRET"))
    if token_id:
        os.environ.setdefault("MODAL_TOKEN_ID", token_id)
    if secret:
        os.environ.setdefault("MODAL_TOKEN_SECRET", secret)
    if secret:
        return ".env"
    if strict and token_id and not os.path.exists(
            os.path.expanduser(os.path.join("~", ".modal.toml"))):
        raise SystemExit(
            "Modal token ID found but MODAL_TOKEN_SECRET is missing.\n"
            "Add it to .env (modal.com -> Settings -> API tokens -> reveal "
            "the as-... secret; create a new token if you lost it),\n"
            "or run: modal setup")
    return "~/.modal.toml"


ensure_modal_auth(strict=False)

app = modal.App("eeg-command-gnn")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.24", "scipy>=1.11", "torch>=2.2")
    .add_local_dir("gnn", remote_path="/root/gnn")
)


def prepare_payload(trials, window_s=2.0, step_s=0.25, mains=50,
                    val_fraction=0.2, seed=0):
    """trials -> (F_tr, y_tr, F_va, y_va, meta). Modal-free, unit-tested."""
    import numpy as np

    from gnn.dataset import build_dataset, trial_split
    from gnn.model import FS, extract_node_features

    labels = sorted({t["label"] for t in trials})
    if len(labels) < 2:
        raise SystemExit("Need trials from at least 2 commands to train.")
    X, y, tids = build_dataset(trials, window_s, step_s, FS)
    tr, va = trial_split(tids, y, val_fraction, seed)
    F = np.stack([extract_node_features(ep, fs=FS, mains=mains) for ep in X])
    F = F.astype(np.float32)
    meta = {"labels": labels, "n_trials": len(trials),
            "n_windows": len(X), "window_s": window_s, "step_s": step_s,
            "mains": mains,
            "train_trials": sorted(set(tids[tr].tolist())),
            "val_trials": sorted(set(tids[va].tolist()))}
    return F[tr], y[tr], F[va], y[va], meta


@app.function(image=image, timeout=1800,
              gpu=os.environ.get("MODAL_GPU") or None)
def fit_remote(F_tr, y_tr, F_va, y_va, hidden=32, epochs=120, batch=16,
               lr=1e-3, seed=0):
    """Fit on a Modal worker. Returns weights + history (pickled)."""
    import sys

    sys.path.insert(0, "/root")
    import torch
    from gnn.model import CHANNELS, CLASSES, N_FEATURES, EEGGraphNet, train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGGraphNet(channels=CHANNELS, fin=N_FEATURES, hidden=hidden,
                        n_classes=len(CLASSES)).to(device)
    history, best = train(model, F_tr, y_tr, F_va, y_va, epochs=epochs,
                          batch=batch, lr=lr, seed=seed)
    n_out = model.head[-1].out_features
    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "history": history, "best_val_acc": best,
            "channels": CHANNELS, "classes": CLASSES, "n_classes": n_out,
            "fin": N_FEATURES, "fs": 250, "hidden": hidden}


@app.local_entrypoint()
def main(data="EEG_recordings", out="gnn/checkpoints", epochs=120, batch=16,
         lr=1e-3, mains=50, window_s=2.0, step_s=0.25, hidden=32,
         val_fraction=0.2, seed=0):
    import json

    import torch

    from gnn import recordings
    from gnn.model import CHANNELS, FS

    print(f"Modal auth via {ensure_modal_auth()}")
    trials = recordings.load_labeled(data, expected_channels=CHANNELS,
                                     expected_fs=FS)
    print(f"Loaded {len(trials)} trials locally, featurizing...")
    F_tr, y_tr, F_va, y_va, meta = prepare_payload(
        trials, window_s, step_s, mains, val_fraction, seed)
    print(f"Windows: train={len(F_tr)} val={len(F_va)}; "
          f"sending {F_tr.nbytes + F_va.nbytes} bytes to Modal...")
    result = fit_remote.remote(F_tr, y_tr, F_va, y_va, hidden=hidden,
                               epochs=epochs, batch=batch, lr=lr, seed=seed)
    os.makedirs(out, exist_ok=True)
    torch.save({k: result[k] for k in (
        "state_dict", "channels", "classes", "n_classes", "fin", "fs",
        "best_val_acc")}, os.path.join(out, "model.pt"))
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump({"channels": result["channels"], "classes": result["classes"],
                   "fs": result["fs"], "fin": result["fin"],
                   "hidden": result["hidden"], "mains": mains,
                   "window_s": window_s, "step_s": step_s,
                   "best_val_acc": result["best_val_acc"],
                   "n_trials": meta["n_trials"], "labels": meta["labels"],
                   "backend": "modal"}, fh, indent=2)
    print(f"Best val acc: {result['best_val_acc']:.3f} -> {out}/")
