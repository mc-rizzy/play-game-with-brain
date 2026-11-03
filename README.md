# play-game-with-brain
Using an EEG and AI model, brain waves can be interpreted to control a video game completely with one's thoughts. No controllers needed.

## Game

Setup and run (from the workspace root):

```
python3 -m venv .venv
.venv/bin/pip install -r game/requirements.txt
.venv/bin/python -m game.main
```

Tests:

```
.venv/bin/python -m unittest discover -s tests
```

Keyboard always works. An AI/EEG model in this workspace can drive the
game simultaneously — same process via `from game import ai_control`
(`ai_control.set_action(move=1.0, jump=True)`), a separate process via
`game/ai_input.json` (see `game/ai_input.example.json`), or the workspace
GNN via `ai_control.EEGModelController`. GNN class labels
`UP/DOWN/LEFT/RIGHT/INTERACT` map to jump/place/move/attack.




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

