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


class ReplaySource:
    """Loop recorded trials as a fake live stream (no hardware needed).

    Streams chunks in real time (``realtime=True``) or as fast as consumed.
    """

    def __init__(self, root="EEG_recordings", labels=None, realtime=True,
                 chunk_s=0.1, expected_channels=None):
        self.root = root
        self.labels = [l.lower() for l in labels] if labels else None
        self.realtime = realtime
        self.chunk_s = chunk_s
        self.expected_channels = expected_channels
        self._lock = threading.Lock()
        self._buf = None  # (C, T) ring of streamed samples
        self._sf = 250
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        from gnn import recordings
        trials = recordings.load_labeled(
            self.root, expected_channels=self.expected_channels)
        if self.labels:
            trials = [t for t in trials if t["label"] in self.labels]
        if not trials:
            raise RuntimeError(f"no trials under {self.root}")
        self._sf = trials[0]["sf"]
        joined = np.concatenate([t["data"] for t in trials], axis=1)
        self._stream = np.ascontiguousarray(joined)
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
        return np.ascontiguousarray(window, dtype=np.float64)


def list_devices():
    """Scan for BrainAccess devices (needs SDK + Bluetooth)."""
    import brainaccess.core as bacore
    bacore.init()
    try:
        return bacore.scan()
    finally:
        bacore.close()
