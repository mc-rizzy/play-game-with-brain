# EEG Model: Brain Access Mini
# Recorded nodes: F3, F4, C3, C4, P3, P4, O1, O2 (8 EEG @ 250 Hz)

# Frontal / Pre-frontal	F3, F4	Frontal theta, eye movement (EOG), blink
# Motor Cortex	C3, C4 (+F3, F4)	Mu (8-13Hz) / Beta (13-30Hz) ERD for left vs right Motor Imagery
# Parietal	P3, P4	Spatial attention, P300, alpha lateralization
# Occipital	O1, O2	Visual cortex, SSVEP, alpha blocking

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt, welch

FS = 250
# Device order, matching the BrainAccess Mini cap + recording meta.
# NOTE: keep this order everywhere (loader, features, adjacency).
CHANNELS = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
N_FEATURES = 12
CLASSES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'INTERACT']

# Live-inference defaults (mirrored in game/ai_control.py, which is torch-free).
WINDOW_S, STEP_S = 2.0, 0.25
EMA, CONFIDENCE, COOLDOWN_S = 0.7, 0.60, 0.5

COORDS_2D = {  # 10-20 approx (x,y); FP* kept for other headsets
    'FP1': (-0.30, 0.95), 'FP2': (0.30, 0.95),
    'F3': (-0.40, 0.60), 'F4': (0.40, 0.60),
    'C3': (-0.55, 0.0), 'C4': (0.55, 0.0),
    'P3': (-0.40, -0.60), 'P4': (0.40, -0.60),
    'O1': (-0.30, -0.95), 'O2': (0.30, -0.95)}
BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)]


def build_static_adj(channels=None, sigma=0.7):
    channels = list(channels or CHANNELS)
    pts = np.array([COORDS_2D[c] for c in channels])
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    A = np.exp(-d ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(A, 1.0)
    return torch.tensor(A, dtype=torch.float32)


def _bandpass_sos(fs=FS):
    return butter(4, [4, 40], btype='bandpass', fs=fs, output='sos')


def _notch_sos(mains=50, fs=FS):
    return butter(4, [mains - 1, mains + 1], btype='bandstop',
                  fs=fs, output='sos')


def preprocess(epoch, fs=FS, mains=50):
    """Bandpass + notch + CAR spatial filter. epoch: (n_ch, T) uV."""
    sos_nt = _notch_sos(mains, fs)
    sos_bp = _bandpass_sos(fs)
    x = sosfiltfilt(sos_nt, epoch, axis=-1)
    x = sosfiltfilt(sos_bp, x, axis=-1)
    x = x - x.mean(axis=0, keepdims=True)  # CAR spatial filter
    return x


def _hjorth(x):
    """Hjorth mobility + complexity of one broadband channel."""
    dx = np.diff(x)
    var_x = np.var(x) + 1e-12
    mobility = np.sqrt(np.var(dx) / var_x)
    ddx = np.diff(dx)
    mob_dx = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12))
    complexity = mob_dx / (mobility + 1e-9)
    return float(mobility), float(complexity)


def extract_node_features(epoch, fs=FS, mains=50):
    """epoch (n_ch, T>=256) -> (n_ch, 12): 5x(log-power, diff-entropy)
    + Hjorth mobility/complexity, standardized across nodes."""
    epoch = np.asarray(epoch, dtype=np.float64)
    n_ch = epoch.shape[0]
    x = preprocess(epoch, fs=fs, mains=mains)
    f, psd = welch(x, fs, nperseg=min(256, x.shape[-1]), axis=-1)
    feats = []
    for ch in range(n_ch):
        row = []
        for lo, hi in BANDS:
            idx = (f >= lo) & (f < hi)
            bp = float(psd[ch, idx].mean()) if idx.any() else 1e-12
            row.append(np.log10(bp + 1e-12))               # log power
            row.append(0.5 * np.log(2 * np.pi * np.e * bp + 1e-12))  # Diff. entropy
        mob, comp = _hjorth(x[ch])
        row += [mob, comp]
        feats.append(row)
    feats = np.array(feats, dtype=np.float32)
    return (feats - feats.mean(0)) / (feats.std(0) + 1e-6)


class GraphConv(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.lin = nn.Linear(fin, fout)
        self.norm = nn.LayerNorm(fout)  # batch-size-1 safe (unlike BatchNorm)

    def forward(self, x, A):  # x:(B,N,F) A:(B,N,N) or (N,N)
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(x.size(0), -1, -1)
        deg = A.sum(-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / torch.sqrt(deg * deg.transpose(1, 2))
        h = torch.bmm(A_norm, self.lin(x))
        return F.elu(self.norm(h))


class EEGGraphNet(nn.Module):
    def __init__(self, n_nodes=None, channels=None, fin=N_FEATURES,
                 hidden=32, n_classes=len(CLASSES), dropout=0.4):
        super().__init__()
        self.channels = list(channels or CHANNELS)
        self.n_nodes = n_nodes or len(self.channels)
        self.register_buffer('A_static',
                             build_static_adj(self.channels))
        # Learnable adjacency (RGNN-style)
        self.E1 = nn.Parameter(torch.randn(self.n_nodes, 8))
        self.E2 = nn.Parameter(torch.randn(self.n_nodes, 8))
        self.gc1 = GraphConv(fin, hidden)
        self.gc2 = GraphConv(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(32, n_classes))

    def forward(self, x):  # (B,N,F)
        A_dyn = F.relu(torch.tanh(self.E1 @ self.E2.T - self.E2 @ self.E1.T))
        A = self.A_static + 0.5 * A_dyn
        h1 = self.drop(self.gc1(x, A))
        h2 = self.gc2(h1, A) + h1  # residual
        pooled = torch.cat([h2.mean(1), h2.max(1)[0]], dim=-1)
        return self.head(pooled)


# --- Training with augmentation for limited data ---
def augment(x, y):  # x:(B,N,F) feature-level augmentation
    if np.random.rand() < 0.5:  # channel dropout - simulate bad electrode
        mask = torch.ones_like(x)
        ch = np.random.randint(0, x.size(1))
        mask[:, ch, :] = 0
        x = x * mask
    x = x + torch.randn_like(x) * 0.05
    return x


def train(model, Xtr, ytr, Xva, yva, epochs=120, batch=16, lr=1e-3,
          seed=0, save_path=None, log_every=10):
    """Train with AdamW + cosine schedule. Keeps the best-val weights.

    Returns (history, best_val_acc). history: list of (epoch, loss, val_acc).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.long)
    Xv = torch.tensor(np.asarray(Xva), dtype=torch.float32)
    yv = torch.tensor(np.asarray(yva), dtype=torch.long)
    best, best_state, history = 0.0, None, []
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        total, nb = 0.0, 0
        for i in range(0, len(Xt), batch):
            b = idx[i:i + batch]
            if len(b) < 2:  # skip singleton tail batch
                continue
            xb = augment(Xt[b].clone(), yt[b]).to(device)
            yb = yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xv.to(device))
            acc = (logits.argmax(1).cpu() == yv).float().mean().item()
        if acc > best:
            best = acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        history.append((ep, total / max(1, nb), acc))
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"Epoch {ep} loss={history[-1][1]:.3f} val_acc={acc:.3f}")
    print(f"Best: {best:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        n_out = model.head[-1].out_features
        classes = (CLASSES if n_out == len(CLASSES)
                   else [f"class_{i}" for i in range(n_out)])
        torch.save({"state_dict": model.state_dict(),
                    "channels": getattr(model, "channels", CHANNELS),
                    "classes": classes, "n_classes": n_out, "fs": FS,
                    "fin": model.gc1.lin.in_features,
                    "best_val_acc": best}, save_path)
        print(f"Saved best model -> {save_path}")
    return history, best


# Usage: X = np.stack([extract_node_features(ep) for ep in epochs])
# model = EEGGraphNet(); train(model, X_train, y_train, X_val, y_val)


def load_checkpoint(path):
    """Load a train.py checkpoint -> (model.eval(), config dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    channels = ckpt.get("channels", CHANNELS)
    n_classes = ckpt.get("n_classes", len(ckpt.get("classes", CLASSES)))
    model = EEGGraphNet(channels=channels, fin=ckpt.get("fin", N_FEATURES),
                        n_classes=n_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"channels": channels, "classes": ckpt.get("classes", CLASSES),
                   "fs": ckpt.get("fs", FS),
                   "best_val_acc": ckpt.get("best_val_acc")}


# Calibration (per player, ~12 mins): Cue arrow on screen 4s, imagine task,
# 2s rest. Collect 30 trials/class = 150 trials. Sliding window gives ~750
# samples. Split 80/20 stratified, never shuffle windows from same trial
# across splits.

# 2s sliding window, step 0.25s, EMA smoothing, confidence gate

# probs_smooth = 0.7*probs_smooth + 0.3*softmax(model(features))
# if probs_smooth.max() > 0.60: # reject uncertain = no move, critical
#     cmd = ['UP','DOWN','LEFT','RIGHT','INTERACT'][probs_smooth.argmax()]
#     send_to_game(cmd); cooldown(0.5) # seconds
