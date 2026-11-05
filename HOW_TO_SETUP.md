# Setup — play-game-with-brain

Using an EEG headset and an AI model, brain waves can be interpreted to
control a video game completely with one's thoughts. No controllers needed.

Runnable entry points: `game/main` (the game), `gnn/viz` (the EEG board).

## Requirements

Run this project on **Windows 10/11, 64-bit (x64)** for the full
experience. The BrainAccess Python library ships native code that only
works there — live headset streaming is Windows-only.

| Platform | Game / training / replay | Live headset |
|---|---|---|
| Windows 64-bit (x64) | ✅ | ✅ (recommended) |
| macOS / Linux (64-bit) | ✅ | ❌ (replay + demo modes only) |
| 32-bit systems (x86 / ARM32) | ❌ | ❌ |
| Headless / embedded Linux ARM | ❌ | ❌ |
| Windows on ARM | ❌ | ❌ |

You need Python 3.12, a BrainAccess Mini headset + Bluetooth for live
mode, and a Hackathon `META_TOKEN` for the Muse calls (the game still
runs without it, minus personalization).

## 1. Game setup (from the workspace root, Windows)

```
py -m venv .venv
.venv\Scripts\pip install -r game/requirements.txt
.venv\Scripts\python -m game.main
```

(On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python`.)

Tests:

```
.venv\Scripts\python -m unittest discover -s tests
```

Keyboard always works. An AI/EEG model in this workspace can drive the
game simultaneously — same process via `from game import ai_control`
(`ai_control.set_action(move=1.0, jump=True)`), a separate process via
`game/ai_input.json` (see `game/ai_input.example.json`), or the workspace
GNN via `ai_control.EEGModelController`. GNN class labels
`UP/DOWN/LEFT/RIGHT/INTERACT` map to jump/place/move/attack.

Controls: A/D or ←/→ move, Space/W jump, 1–9/0 + wheel hotbar, E craft,
F torch, C brain calibration, F11 fullscreen, LMB mine/attack, RMB place.

## 2. EEG dependencies + BrainAccess SDK

```
.venv\Scripts\pip install -r gnn/requirements.txt
cd BrainAccessSDK-windows/python_api
pip install .
```

Verify installation:

```
python -c "import brainaccess; print('BrainAccess successfully installed!')"
```

Live streaming additionally needs `pip install mne` and Bluetooth.
(The `BrainAccessSDK-linux/` folder in this workspace is outdated and
only kept as an API reference.)

## 3. API keys (`.env` in the workspace root)

```
META_TOKEN=...            # Meta Model API key (Muse calls)
MODAL_TOKEN_ID=...        # only for Modal cloud training
MODAL_TOKEN_SECRET=...    # (or run `modal setup` instead)
```

## 4. EEG pipeline (BrainAccess Mini, 250 Hz)

1. **Record**: put one `*_eeg.db` trial (+ `-wal`/`-shm` sidecars) per
   `EEG_recordings/<up|down|left|right|interact>/sub-*/ses-*/` folder
   (~30 trials/command; folder name is the label).
2. **Train** — pick one (same data, same `model.pt` output):
   - Local device (free, uses your own CPU/GPU):
     `.venv\Scripts\python -m gnn.train --data EEG_recordings --out gnn/checkpoints`
   - Modal cloud (uses credits; `.venv\Scripts\pip install modal` first):
     `.venv\Scripts\modal run gnn/modal_train.py --data EEG_recordings --out gnn/checkpoints`
     (add `set MODAL_GPU=T4 &&` in front for a GPU worker).
   Add `--fast` to either for the low-latency 0.5 s rolling-PSD model
   (use a separate `--out` folder so it doesn't overwrite the classic
   one). Both do a trial-wise 80/20 split and save `model.pt` +
   `config.json`. Out of Modal credits? Just run the local command.
3. **Calibrate** each new player (15 s): `.venv\Scripts\python -m
   game.calibrate --source live --device "..." --model
   gnn/checkpoints/model.pt` — or just press **C** inside the game.
   The wizard flashes 5 cues, few-shot adapts the GNN head to that
   brain (MAML-style), asks Muse to coach the player + tune the live
   gates, and saves `adapter.pt` + `calibration_report.json`.
   (`--source demo` runs with no hardware.)
4. **Watch**: `.venv\Scripts\python -m gnn.viz --source replay`
   (model auto-loaded; live board: raw/CAR traces, band powers, GNN
   feature heatmap, prediction, spectrum, contacts, battery, accel;
   `--source live --device "..."` on hardware; flip `USE_FAST_MODEL`
   in `gnn/viz.py` for the fast checkpoint; `--fullscreen`, F11).
5. **Play**: run `.venv\Scripts\python -m game.main` and
   `.venv\Scripts\python -m gnn.live --source live --device "..."
   --model gnn/checkpoints/model.pt --adapter gnn/checkpoints/adapter.pt`
   in two terminals — brainwaves steer the game alongside the
   keyboard. Add `--muse-fallback` so Muse referees uncertain frames.
   No headset? `--source replay` works everywhere.

## 5. Architecture: dual-layer meta-pipeline

Raw EEG straight into a GNN means ~4000 non-stationary floats per step —
high latency, unstable training. Instead:

1. **Feature Extraction Layer** (`gnn/features.py`): BrainAccess SDK
   stream → rolling 0.5 s chunks → Welch PSD → theta/alpha/beta/gamma
   log-power + relative power (8 node features @ 2 Hz, streaming
   filters). Train this path with `gnn/train.py --fast`.
2. **Spatial GNN Layer** (`gnn/graph.py`): nodes = cap electrodes,
   edges = spatial proximity (+ learned affinities) blended with
   per-frame functional connectivity (Pearson correlation) →
   spatial-temporal latent vector. Functional edges engage after wizard
   adaptation tunes the head with them, so inference always matches
   what the head was trained on.
3. **Meta-model Layer** (`gnn/adapt.py` + `game/calibrate.py`): the 15 s
   Calibration Wizard collects a labeled support set and fine-tunes the
   head only (MAML-style inner loop) — the game instantly understands
   each new brain.

## 6. Meta models (integral, not decorative)

Every demo run calls Meta's hosted **Muse Spark** (`muse-spark-1.3`)
through the Meta Model API (`https://api.meta.ai/v1`, see `gnn/muse.py`):

- **Calibration coach**: Muse reads the player's fresh 15 s EEG summary
  (band powers, class separability, adaptation fit) and returns personal
  coaching + tuned confidence/cooldown/EMA gates.
- **Intent referee** (`--muse-fallback`): when the GNN is in the gray
  zone, Muse disambiguates intent from recent predictions + game context
  instead of dropping the input.

Meta's models also built the project — game, signal-processing research,
GNN architecture, training workflows, model↔game interface, dashboard,
tests, docs (full list in [README.md](README.md)). Every runtime call is
timed, printed, and logged into `calibration_report.json` (`muse.calls`),
so the demo video + repo both show the Meta model working.

## Troubleshooting

- **Visualizer shows NO MODEL / replay finds no trials**: you launched
  from the wrong folder — run from the workspace root (paths fall back
  to it automatically, but the terminal location still matters for `-m`).
- **Prediction never changes**: pass a trained `--model` (or rely on the
  auto-loaded default) and give replay ~30 s — it cycles all commands.
- **`modal run` can't connect**: Modal needs direct network access;
  corporate VPNs/proxies may block it — train locally instead.
