# AGENTS.md

## Commands

- Run the game: `.venv\Scripts\python -m game.main` (from root on Windows; needs `pip install -r game/requirements.txt`)
- Run the full test suite: `.venv\Scripts\python -m unittest discover -s tests`
- Regenerate pixel art: `.venv\Scripts\python game/generate_assets.py` (writes `game/assets/*.png`)
- Train EEG model: `.venv\Scripts\python -m gnn.train --data EEG_recordings --out gnn/checkpoints`
- EEG visualizer: `.venv\Scripts\python -m gnn.viz --source replay --model gnn/checkpoints/model.pt`
- Brain bridge: `.venv\Scripts\python -m gnn.live --source live --device "..." --model gnn/checkpoints/model.pt`

## Notes

- Single package-style imports (`from game.x import ...`); run via `-m` or `game/main.py` bootstrap.
- `game/ai_input.json` (live AI input, gitignored) merges with keyboard input; see `game/ai_input.example.json`.
