# EEG recordings (BrainAccess Mini, 250 Hz)

One folder per game command. The folder name IS the label, so record each
command's folder while imagining only that command.

```
EEG_recordings/
  left/  right/  up/  down/  interact/
    sub-001/
      ses-001/
        *_eeg.db  (+ .db-wal / .db-shm sidecars — keep all three together)
```

Rules:

- One trial = one `*_eeg.db` file (plus its `-wal`/`-shm` sidecars). Copy all
  three files; the signal lives mostly in the `-wal` file.
- Aim for ~30 trials per command (~4 s imagining + 2 s rest each). More
  subjects/sessions help: add `sub-002/`, `ses-002/`, ... with the same layout.
- The trainer splits whole trials into train/val (never windows from one trial
  on both sides) — see `gnn/train.py`.
- Channel order expected: F3 F4 C3 C4 P3 P4 O1 O2 (+ contact/accel/misc, ignored).
