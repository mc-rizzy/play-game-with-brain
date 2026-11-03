# EEG Model: Brain Access Mini
# Nodes of data collected: P1, FP2, F3, F4, C3, C4, P3, P4, O1, O2

# Frontal / Pre-frontal	FP1, FP2, F3, F4	Eye movement (EOG), blink, frontal theta, SSVEP attention
# Motor Cortex	C3, C4, F3, F4	Mu (8-13Hz) / Beta (13-30Hz) ERD for left vs right hand Motor Imagery
# Parietal	P3, P4	Spatial attention, P300, alpha lateralization
# Occipital	O1, O2	Visual cortex, SSVEP, alpha blocking




import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt, welch

FS, CHANNELS = 250, ['FP2','FP1','F4','F3','C3','C4','P3','P4','O1','O2']
# NOTE: keep this order everywhere
COORDS_2D = { # 10-20 approx (x,y)
 'FP1':(-0.30,0.95),'FP2':(0.30,0.95),'F3':(-0.40,0.60),'F4':(0.40,0.60),
 'C3':(-0.55,0.0),'C4':(0.55,0.0),'P3':(-0.40,-0.60),'P4':(0.40,-0.60),
 'O1':(-0.30,-0.95),'O2':(0.30,-0.95)}
BANDS = [(4,8),(8,13),(13,20),(20,30),(30,40)]

def build_static_adj(sigma=0.7):
    pts = np.array([COORDS_2D[c] for c in CHANNELS])
    d = np.linalg.norm(pts[:,None]-pts[None,:],axis=-1)
    A = np.exp(-d**2/(2*sigma**2)); np.fill_diagonal(A,1.0)
    return torch.tensor(A, dtype=torch.float32)

def preprocess(epoch, fs=FS): # epoch: (10, T) uV
    sos_bp = butter(4,[4,40],btype='bandpass',fs=fs,output='sos')
    sos_nt = butter(4,[49,51],btype='bandstop',fs=fs,output='sos')
    x = sosfiltfilt(sos_nt, epoch, axis=-1)
    x = sosfiltfilt(sos_bp, x, axis=-1)
    x = x - x.mean(axis=0, keepdims=True) # CAR spatial filter
    return x

def extract_node_features(epoch, fs=FS): # -> (10, 12)
    x = preprocess(epoch)
    f, psd = welch(x, fs, nperseg=256, axis=-1) # (10, F, )
    feats = []
    for ch in range(10):
        row=[]
        for lo,hi in BANDS:
            idx=(f>=lo)&(f<hi); bp=psd[ch,idx].mean()
            row.append(np.log10(bp+1e-12))               # log power
            row.append(0.5*np.log(2*np.pi*np.e*bp+1e-12)) # Differential Entropy
        # Hjorth on broadband
        dx=np.diff(x[ch]); row += [np.std(dx)/(np.std(x[ch])+1e-9), 0.0]
        feats.append(row)
    feats=np.array(feats, dtype=np.float32)
    return (feats - feats.mean(0)) / (feats.std(0)+1e-6)

class GraphConv(nn.Module):
    def __init__(self, fin, fout):
        super().__init__(); self.lin=nn.Linear(fin,fout); self.bn=nn.BatchNorm1d(fout)
    def forward(self, x, A): # x:(B,N,F) A:(B,N,N) or (N,N)
        if A.dim()==2: A=A.unsqueeze(0)
        deg=A.sum(-1,keepdim=True).clamp(min=1e-6)
        A_norm=A/torch.sqrt(deg*deg.transpose(1,2))
        h=torch.bmm(A_norm, self.lin(x))
        return F.elu(self.bn(h.transpose(1,2)).transpose(1,2))

class EEGGraphNet(nn.Module):
    def __init__(self, n_nodes=10, fin=12, hidden=32, n_classes=5, dropout=0.4):
        super().__init__()
        self.register_buffer('A_static', build_static_adj())
        # Learnable adjacency (RGNN-style)
        self.E1=nn.Parameter(torch.randn(n_nodes,8)); self.E2=nn.Parameter(torch.randn(n_nodes,8))
        self.gc1=GraphConv(fin,hidden); self.gc2=GraphConv(hidden,hidden)
        self.drop=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(hidden*2,32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32,n_classes))
    def forward(self, x): # (B,10,12)
        A_dyn = F.relu(torch.tanh(self.E1 @ self.E2.T - self.E2 @ self.E1.T)) # anti-symmetric
        A = self.A_static + 0.5*A_dyn
        h1=self.drop(self.gc1(x,A)); h2=self.gc2(h1,A)+h1 # residual
        pooled=torch.cat([h2.mean(1), h2.max(1)[0]], dim=-1)
        return self.head(pooled)

# --- Training with augmentation for limited data ---
def augment(x, y): # x:(B,N,F) feature-level augmentation
    if np.random.rand()<0.5: # channel dropout - simulate bad electrode
        mask=torch.ones_like(x); ch=np.random.randint(0,x.size(1)); mask[:,ch,:]=0; x=x*mask
    x = x + torch.randn_like(x)*0.05
    return x

def train(model, Xtr, ytr, Xva, yva, epochs=120):
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt,yt=torch.tensor(Xtr),torch.tensor(ytr); Xv,yv=torch.tensor(Xva),torch.tensor(yva)
    best=0
    for ep in range(epochs):
        model.train(); idx=torch.randperm(len(Xt))
        for i in range(0,len(Xt),16):
            b=idx[i:i+16]; xb=augment(Xt[b].clone(), yt[b])
            opt.zero_grad(); loss=loss_fn(model(xb), yt[b]); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): acc=(model(Xv).argmax(1)==yv).float().mean().item()
        best=max(best,acc); print(f"Epoch {ep} val_acc={acc:.3f}")
    print(f"Best: {best:.3f}")

# Usage: X = np.stack([extract_node_features(ep) for ep in epochs]) # epochs from BrainAccess SDK stream
# model = EEGGraphNet(); train(model, X_train, y_train, X_val, y_val)





# Calibration (per player, ~12 mins): Cue arrow on screen 4s, imagine task, 2s rest. Collect 30 trials/class = 150 trials. Sliding window gives ~750 samples. Split 80/20 stratified, never shuffle windows from same trial across splits.

# 2s sliding window, step 0.25s, EMA smoothing, confidence gate

# probs_smooth = 0.7*probs_smooth + 0.3*softmax(model(features))
# if probs_smooth.max() > 0.60: # reject uncertain = no move, critical for games
#     cmd = ['UP','DOWN','LEFT','RIGHT','INTERACT'][probs_smooth.argmax()]
#     send_to_game(cmd); cooldown(0.5) # seconds