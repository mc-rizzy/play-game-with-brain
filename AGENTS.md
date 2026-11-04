# AGENTS.md

## Commands

- Run the game: `.venv\Scripts\python -m game.main` (from root on Windows; needs `pip install -r game/requirements.txt`)
- Run the full test suite: `.venv\Scripts\python -m unittest discover -s tests`
- Regenerate pixel art: `.venv\Scripts\python game/generate_assets.py` (writes `game/assets/*.png`)
- Train EEG model locally: `.venv\Scripts\python -m gnn.train --data EEG_recordings --out gnn/checkpoints`
- Train EEG model on Modal: `.venv\Scripts\modal run gnn/modal_train.py --data EEG_recordings --out gnn/checkpoints` (needs `pip install modal` + `modal setup`; `MODAL_GPU=T4` for GPU)
- EEG visualizer: `.venv\Scripts\python -m gnn.viz --source replay --model gnn/checkpoints/model.pt`
- Brain bridge: `.venv\Scripts\python -m gnn.live --source live --device "..." --model gnn/checkpoints/model.pt`
- Calibrate a new brain (15 s wizard): `.venv\Scripts\python -m game.calibrate --source live --device "..." --model gnn/checkpoints/model.pt` (or press C in-game; `--source demo` needs no hardware)
- Bridge with personal adapter: add `--adapter gnn/checkpoints/adapter.pt` (wizard output); add `--muse-fallback` for Muse gray-zone referee
- Train low-latency model: `.venv\Scripts\python -m gnn.train --fast --data EEG_recordings --out gnn/checkpoints` (0.5 s rolling-PSD 8-dim feats)
- Muse live check: `.venv\Scripts\python -c "from gnn.muse import MuseClient; print(MuseClient().ping())"` (needs `META_TOKEN` in `.env`)

## Notes

- Single package-style imports (`from game.x import ...`); run via `-m` or `game/main.py` bootstrap.
- `game/ai_input.json` (live AI input, gitignored) merges with keyboard input; see `game/ai_input.example.json`.
