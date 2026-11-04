# play-game-with-brain
Using an EEG and AI model, brain waves can be interpreted to control a video game completely with one's thoughts. No controllers needed.

Runnable:
game/main
gnn/viz

## Game

Setup and run (from the workspace root, Windows):

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

## EEG pipeline (BrainAccess Mini, 250 Hz)

Install ML deps: `.venv\Scripts\pip install -r gnn/requirements.txt`

1. Record: put one `*_eeg.db` trial (+ `-wal`/`-shm`) per
   `EEG_recordings/<up|down|left|right|interact>/sub-*/ses-*/` folder
   (~30 trials/command; folder name is the label).
2. Train — pick one (same data, same `model.pt` output):
   - Local device (free, uses your own CPU/GPU):
     `.venv\Scripts\python -m gnn.train --data EEG_recordings --out gnn/checkpoints`
   - Modal cloud (faster, uses credits; `.venv\Scripts\pip install modal`,
     then auth via `.venv\Scripts\modal setup`, or a `.env` file with
     `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` from modal.com API tokens):
     `.venv\Scripts\modal run gnn/modal_train.py --data EEG_recordings --out gnn/checkpoints`
     (add `set MODAL_GPU=T4 &&` in front for a GPU worker).
   Both do a trial-wise 80/20 split and save `model.pt` + `config.json`.
   Out of Modal credits? Just run the local command instead.
3. Calibrate each new player (15 s): `.venv\Scripts\python -m
   game.calibrate --source live --device "..." --model
   gnn/checkpoints/model.pt` — or just press **C** inside the game.
   The wizard flashes 5 cues, few-shot adapts the GNN head to that
   brain (MAML-style), asks Muse to coach the player + tune the live
   gates, and saves `adapter.pt` + `calibration_report.json`.
   (`--source demo` runs with no hardware.)
4. Watch: `.venv\Scripts\python -m gnn.viz --source replay`
   (model auto-loaded; live board: raw/CAR traces, band powers, GNN
   feature heatmap, prediction, spectrum, contacts, battery, accel;
   `--source live --device "..."` on hardware; flip `USE_FAST_MODEL`
   in `gnn/viz.py` for the fast checkpoint).
5. Play: run `.venv\Scripts\python -m game.main` and
   `.venv\Scripts\python -m gnn.live --source live --device "..."
   --model gnn/checkpoints/model.pt --adapter gnn/checkpoints/adapter.pt`
   in two terminals — brainwaves steer the game alongside the
   keyboard. Add `--muse-fallback` so Muse referees uncertain frames.

## Dual-layer meta-pipeline

Raw EEG straight into a GNN means ~4000 non-stationary floats per step —
high latency, unstable training. Instead:

1. **Feature Extraction Layer** (`gnn/features.py`): BrainAccess SDK
   stream -> rolling 0.5 s chunks -> Welch PSD -> theta/alpha/beta/gamma
   log-power + relative power (8 node features @ 2 Hz, streaming filters).
   Train this path with `gnn/train.py --fast`.
2. **Spatial GNN Layer** (`gnn/graph.py`): nodes = cap electrodes,
   edges = spatial proximity (+ learned affinities) blended with
   per-frame functional connectivity (Pearson correlation) ->
   spatial-temporal latent vector. Functional edges engage after wizard
   adaptation tunes the head with them, so inference always matches
   what the head was trained on.
3. **Meta-model Layer** (`gnn/adapt.py` + `game/calibrate.py`): the 15 s
   Calibration Wizard collects a labeled support set and fine-tunes the
   head only (MAML-style inner loop) — the game instantly understands
   each new brain.

## Meta models (integral, not decorative)

Every demo run calls Meta's hosted **Muse Spark** (`muse-spark-1.3`)
through the Meta Model API (`https://api.meta.ai/v1`, see `gnn/muse.py`):

- **Calibration coach**: Muse reads the player's fresh 15 s EEG summary
  (band powers, class separability, adaptation fit) and returns personal
  coaching + tuned confidence/cooldown/EMA gates.
- **Intent referee** (`--muse-fallback`): when the GNN is in the gray
  zone, Muse disambiguates intent from recent predictions + game context
  instead of dropping the input.

Setup: put the hackathon key in `.env` as `META_TOKEN=...` (or export
`MODEL_API_KEY`). Every call is timed, printed, and logged into
`calibration_report.json` (`muse.calls`), so the demo video + repo both
show the Meta model working. Without a key/network the pipeline falls
back to heuristic gates — it still plays, minus the personalization.

Live streaming needs Windows + `mne` + the BrainAccess Windows SDK
(`pip install mne` plus the SDK's `python_api`); replay mode needs
no hardware. (The `BrainAccessSDK-linux/` folder in this workspace is
outdated and only kept as an API reference.)




# Hackathon Submisison to Global Meta Hackathon 2026:
Team name: Team 01f-w8c9spo





# Set up

set up the venv first.

Then install the following:

pip install pygame
cd BrainAccessSDK-windows/python_api
pip install .

Verify installation:
python -c "import brainaccess; print('BrainAccess successfully installed!')"

