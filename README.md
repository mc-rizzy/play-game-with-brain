# play-game-with-brain
Using an EEG and AI model, brain waves can be interpreted to control a video game completely with one's thoughts. No controllers needed.

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
2. Train: `.venv\Scripts\python -m gnn.train --data EEG_recordings
   --out gnn/checkpoints` (trial-wise 80/20 split, saves `model.pt`).
3. Watch: `.venv\Scripts\python -m gnn.viz --source replay
   --model gnn/checkpoints/model.pt` (raw traces + band powers +
   live prediction; `--source live --device "..."` on hardware).
4. Play: run `.venv\Scripts\python -m game.main` and
   `.venv\Scripts\python -m gnn.live --source live --device "..."
   --model gnn/checkpoints/model.pt` in two terminals — brainwaves
   steer the game alongside the keyboard.

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

