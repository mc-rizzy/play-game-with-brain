# AGENTS.md

## Commands

- Run the game: `.venv/bin/python -m game.main` (from root; needs `pip install -r game/requirements.txt`)
- Run the full test suite: `.venv/bin/python -m unittest discover -s tests`
- Regenerate pixel art: `.venv/bin/python game/generate_assets.py` (writes `game/assets/*.png`)

## Notes

- Single package-style imports (`from game.x import ...`); run via `-m` or `game/main.py` bootstrap.
- `game/ai_input.json` (live AI input, gitignored) merges with keyboard input; see `game/ai_input.example.json`.
