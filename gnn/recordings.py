"""Load BrainAccess Mini recordings from EEG_recordings/.

Layout (folder name = label)::

    EEG_recordings/<command>/sub-*/ses-*/*_eeg.db (+ -wal/-shm sidecars)

Each ``*_eeg.db`` is one trial: SQLite tables ``data_<uuid>`` (numpy
float64 chunks = 22 channels x 25-50 samples @ 250 Hz),
``meta_<uuid>`` (channel names/types/units, sampling rate) and
``annotations_<uuid>``. Only channels typed EEG are returned, reordered to
``model.CHANNELS``.
"""

import io
import os
import sqlite3
import warnings

COMMANDS = ("up", "down", "left", "right", "interact")


def _table(con, prefix):
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    hits = [n for n in names if n.startswith(prefix)]
    if not hits:
        raise ValueError(f"no {prefix}* table found")
    return hits[0]


def find_trials(root="EEG_recordings"):
    """Return [(label, db_path)] sorted; label = top folder name."""
    trials = []
    if not os.path.isdir(root):
        return trials
    for command in sorted(os.listdir(root)):
        if command.lower() not in COMMANDS:
            continue
        cdir = os.path.join(root, command)
        if not os.path.isdir(cdir):
            continue
        for dirpath, _dirs, files in os.walk(cdir):
            for fn in sorted(files):
                if not fn.endswith("_eeg.db"):
                    continue
                trials.append((command.lower(),
                               os.path.join(dirpath, fn)))
    return sorted(trials)


def load_trial(path, expected_channels=None, expected_fs=250):
    """Load one trial -> dict(data (n_eeg, T) float64, sf, channels, ...).

    Opens read-only; warns when -wal/-shm sidecars are missing (the signal
    usually lives in the WAL) or timestamps show gaps.
    """
    base = os.path.splitext(path)[0]
    for ext in ("-wal", "-shm"):
        if not os.path.exists(path + ext):
            warnings.warn(f"{path}: sidecar {os.path.basename(path)}{ext} "
                          f"missing, data may be incomplete")
    from pathlib import Path
    uri = Path(os.path.abspath(path)).as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        data_t = _table(con, "data_")
        meta_t = _table(con, "meta_")
        names, types, _units, sf, _id = con.execute(
            f'SELECT channels, channels_type, channels_unit, sf, id '
            f'FROM "{meta_t}" LIMIT 1').fetchone()
        names = names.split(",")
        types = types.split(",")
        eeg_idx = [i for i, t in enumerate(types) if t.strip() == "EEG"]
        eeg_names = [names[i].strip() for i in eeg_idx]
        if expected_channels:
            want = list(expected_channels)
            missing = [c for c in want if c not in eeg_names]
            if missing:
                raise ValueError(f"{path}: missing EEG channels {missing}")
            order = [eeg_idx[eeg_names.index(c)] for c in want]
        else:
            order = eeg_idx
            want = eeg_names
        if abs(float(sf) - expected_fs) > 1e-6:
            warnings.warn(f"{path}: sf={sf}, expected {expected_fs}")
        chunks, times = [], []
        for blob, tblob, _clk in con.execute(
                f'SELECT data, time, local_clock FROM "{data_t}" ORDER BY rowid'):
            import numpy as np
            chunks.append(np.load(io.BytesIO(blob), allow_pickle=False))
            times.append(np.load(io.BytesIO(tblob), allow_pickle=False))
    finally:
        con.close()
    import numpy as np
    if not chunks:
        raise ValueError(f"{path}: no data chunks")
    # Chunk widths vary in the wild (25 and 50 samples observed).
    picked = [c[order, :] for c in chunks]
    data = np.concatenate(picked, axis=1)
    chunk_samples = [c.shape[1] for c in picked]
    time = np.concatenate(times)
    gaps = np.diff(time)
    step = 1.0 / float(sf)
    if len(gaps) and (gaps.max() > 3 * step or gaps.min() <= 0):
        warnings.warn(f"{path}: timestamp gaps detected "
                      f"(min={gaps.min():.4f}s max={gaps.max():.4f}s)")
    return {"data": np.ascontiguousarray(data), "sf": float(sf),
            "channels": want, "time": time, "path": path,
            "n_chunks": len(chunks), "chunk_samples": chunk_samples}


def load_labeled(root="EEG_recordings", expected_channels=None,
                 expected_fs=250):
    """Load all trials -> list of dicts with label + trial_id.

    trial_id (relative db path) groups windows for trial-wise splits.
    """
    out = []
    for label, path in find_trials(root):
        trial = load_trial(path, expected_channels, expected_fs)
        trial["label"] = label
        trial["trial_id"] = os.path.relpath(path, root)
        out.append(trial)
    return out
