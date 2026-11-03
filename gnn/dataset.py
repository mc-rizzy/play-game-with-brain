"""Windowing + trial-wise splits for EEG command data.

One window = 2 s @ 250 Hz stepped every 0.25 s. Splits are stratified by
trial label and keep all windows of a trial on one side (no leakage).
"""

import numpy as np

CLASSES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'INTERACT']
LABEL_TO_INDEX = {c.lower(): i for i, c in enumerate(CLASSES)}


def sliding_windows(data, window=500, step=62):
    """(C, T) -> (n_win, C, window); drops the trailing partial window."""
    data = np.asarray(data)
    n_ch, total = data.shape
    if total < window:
        return np.zeros((0, n_ch, window), dtype=data.dtype)
    starts = range(0, total - window + 1, step)
    return np.stack([data[:, s:s + window] for s in starts])


def build_dataset(trials, window_s=2.0, step_s=0.25, sf=250):
    """trials: load_labeled() dicts -> (X, y, trial_ids).

    X: (n_win, C, T) float32 epochs, y: (n_win,) int64 class ids.
    """
    window = int(round(window_s * sf))
    step = int(round(step_s * sf))
    Xs, ys, tids = [], [], []
    for trial in trials:
        label = trial["label"].lower()
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"unknown label {label!r} in {trial['path']}")
        wins = sliding_windows(trial["data"], window, step)
        if len(wins) == 0:
            continue
        Xs.append(wins)
        ys.append(np.full(len(wins), LABEL_TO_INDEX[label], dtype=np.int64))
        tids.extend([trial["trial_id"]] * len(wins))
    if not Xs:
        raise ValueError("no windows: recordings too short or empty")
    X = np.concatenate(Xs).astype(np.float32)
    return X, np.concatenate(ys), np.array(tids)


def trial_split(trial_ids, y, val_fraction=0.2, seed=0):
    """Stratified trial split -> (train_idx, val_idx) over windows."""
    rng = np.random.RandomState(seed)
    trial_ids = np.asarray(trial_ids)
    y = np.asarray(y)
    uniq = sorted(set(trial_ids.tolist()))
    by_label = {}
    for tid in uniq:
        idx = np.where(trial_ids == tid)[0]
        lab = int(np.bincount(y[idx]).argmax())
        by_label.setdefault(lab, []).append(tid)
    train_t, val_t = [], []
    for _lab, tids in by_label.items():
        rng.shuffle(tids)
        n_val = max(1 if len(tids) > 1 else 0,
                    int(round(len(tids) * val_fraction)))
        n_val = min(n_val, len(tids) - 1) if len(tids) > 1 else 0
        val_t.extend(tids[:n_val])
        train_t.extend(tids[n_val:])
    val_set = set(val_t)
    is_val = np.array([t in val_set for t in trial_ids])
    return np.where(~is_val)[0], np.where(is_val)[0]
