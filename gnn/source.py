"""EEG stream sources: live BrainAccess Mini or recording replay.

Both expose the same interface so the visualizer and live player run
unchanged on hardware or on recorded files::

    source = LiveSource(device_name)   # needs brainaccess + mne + hardware
    source = ReplaySource("EEG_recordings", realtime=True)
    with source:
        while True:
            window = source.read_window(seconds=2.0)  # (n_ch, T) or None
"""

import threading
import time

import numpy as np


class LiveSource:
    """Rolling 2 s buffer from a BrainAccess Mini (Windows + SDK + mne)."""

    def __init__(self, device_name, sfreq=250, cap=None):
        self.device_name = device_name
        self.sfreq = sfreq
        self.cap = cap  # None -> SDK default 8-ch Mini layout
        self._eeg = None
        self._mgr = None

    def __enter__(self):
        try:
            from brainaccess.utils.acquisition import EEG
            from brainaccess.core.eeg_manager import EEGManager
        except ImportError as exc:
            raise RuntimeError(
                "LiveSource needs the BrainAccess Windows SDK + mne. "
                "Use ReplaySource without hardware.") from exc
        self._mgr = EEGManager()
        self._eeg = EEG(mode="roll")
        kwargs = {"sfreq": self.sfreq}
        if self.cap is not None:
            kwargs["cap"] = self.cap
        self._eeg.setup(self._mgr, self.device_name, **kwargs)
        self._eeg.start_acquisition()
        return self

    def __exit__(self, *exc):
        try:
            if self._eeg is not None:
                self._eeg.stop_acquisition()
            if self._mgr is not None:
                self._mgr.disconnect()
        finally:
            if self._eeg is not None:
                self._eeg.close()
            self._eeg = self._mgr = None

    def read_window(self, seconds=2.0, channels=None):
        raw = self._eeg.get_mne(tim=seconds, annotations=False)
        data, _times = raw.get_data(return_times=True)
        names = raw.info["ch_names"]
        if channels:
            idx = [names.index(c) for c in channels]
            data = data[idx, :]
        return np.ascontiguousarray(data, dtype=np.float64)

    def read_battery(self):
        """Headset battery percent (None when unavailable)."""
        try:
            return int(self._eeg.get_battery())
        except Exception:
            return None

    def read_accel(self):
        """Latest (x, y, z) accelerometer sample in m/s2 (None if absent)."""
        try:
            raw = self._eeg.get_mne(tim=0.2, annotations=False)
            data, _times = raw.get_data(return_times=True)
            names = raw.info["ch_names"]
            idx = [names.index(c) for c in ("Accel_x", "Accel_y", "Accel_z")]
            return tuple(float(v) for v in data[idx, -1])
        except Exception:
            return None

    def read_contacts(self, channels=None):
        """Contact quality is not in the live stream -> None.

        Run the SDK impedance-measurement mode for live contact values;
        recordings carry contact_* columns instead (see ReplaySource).
        """
        return None


def interleave_trials(trials):
    """Round-robin trials across labels (deterministic, sorted labels).

    Replay then cycles UP/DOWN/LEFT/... like a live session instead of
    sitting on one command for minutes, so dashboards and demos show
    genuinely changing predictions.
    """
    groups = {}
    for trial in trials:
        groups.setdefault(trial["label"], []).append(trial)
    queues = [list(groups[key]) for key in sorted(groups)]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


class ReplaySource:
    """Loop recorded trials as a fake live stream (no hardware needed).

    Streams chunks in real time (``realtime=True``) or as fast as consumed.
    Trials replay round-robin across labels (``interleave=True``) so the
    stream cycles commands; pass ``interleave=False`` for label-grouped
    (sorted) order.
    """

    def __init__(self, root="EEG_recordings", labels=None, realtime=True,
                 chunk_s=0.1, expected_channels=None, interleave=True):
        self.root = root
        self.labels = [l.lower() for l in labels] if labels else None
        self.realtime = realtime
        self.chunk_s = chunk_s
        self.expected_channels = expected_channels
        self.interleave = interleave
        self._lock = threading.Lock()
        self._buf = None  # (C, T) ring of streamed samples
        self._sf = 250
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        from gnn import recordings
        trials = recordings.load_labeled(
            self.root, expected_channels=self.expected_channels, full=True)
        if self.labels:
            trials = [t for t in trials if t["label"] in self.labels]
        if self.interleave:
            trials = interleave_trials(trials)
        if not trials:
            raise RuntimeError(f"no trials under {self.root}")
        self._sf = trials[0]["sf"]
        self._names = trials[0]["full_names"]
        joined = np.concatenate([t["full_data"] for t in trials], axis=1)
        self._stream = np.ascontiguousarray(joined)
        eeg_rows = [self._names.index(c) for c in trials[0]["channels"]]
        self._eeg_rows = eeg_rows
        self._eeg_names = list(trials[0]["channels"])
        cap = int(self._sf * 10)
        self._buf = np.zeros((self._stream.shape[0], 0))
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _pump(self):
        n = int(self._sf * self.chunk_s)
        pos = 0
        total = self._stream.shape[1]
        while not self._stop.is_set():
            nxt = pos + n
            if nxt >= total:
                piece = np.concatenate(
                    [self._stream[:, pos:], self._stream[:, :nxt - total]],
                    axis=1)
                pos = nxt - total
            else:
                piece = self._stream[:, pos:nxt]
                pos = nxt
            with self._lock:
                self._buf = np.concatenate([self._buf, piece], axis=1)
                self._buf = self._buf[:, -self._cap:]
            if self.realtime:
                time.sleep(self.chunk_s)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def sf(self):
        return self._sf

    @property
    def buffered_seconds(self):
        with self._lock:
            return self._buf.shape[1] / self._sf

    def read_window(self, seconds=2.0, channels=None):
        want = int(seconds * self._sf)
        with self._lock:
            if self._buf.shape[1] < want:
                return None
            window = self._buf[:, -want:].copy()
        if channels:
            rows = [self._names.index(c) for c in channels]
        else:
            rows = self._eeg_rows
        return np.ascontiguousarray(window[rows, :], dtype=np.float64)

    def _latest(self, name):
        with self._lock:
            if self._buf.shape[1] == 0 or name not in self._names:
                return None
            return float(self._buf[self._names.index(name), -1])

    def read_battery(self):
        """Latest Battery value in percent (None when absent)."""
        value = self._latest("Battery")
        return None if value is None else int(round(value))

    def read_accel(self):
        """Latest (x, y, z) accelerometer sample in m/s2 (None if absent)."""
        xyz = tuple(self._latest(c)
                    for c in ("Accel_x", "Accel_y", "Accel_z"))
        return None if any(v is None for v in xyz) else xyz

    def read_contacts(self, channels=None):
        """Latest contact_* values as {channel: value} (None if absent)."""
        channels = list(channels or self._eeg_names)
        out = {}
        for ch in channels:
            for candidate in (f"contact_{ch}", f"contact{ch}", ch):
                value = self._latest(candidate)
                if value is not None:
                    out[ch] = value
                    break
        return out or None


def list_devices():
    """Scan for BrainAccess devices (needs SDK + Bluetooth)."""
    import brainaccess.core as bacore
    bacore.init()
    try:
        return bacore.scan()
    finally:
        bacore.close()
