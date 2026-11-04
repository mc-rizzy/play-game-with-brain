"""Feature Extraction Layer: rolling 0.5 s PSD chunks from the EEG stream.

Why this layer exists: streaming raw samples straight into a GNN means
~4000 non-stationary floats per inference step (8 ch x 250 Hz x 2 s),
which causes high latency and unstable training. This layer compresses
the stream into compact, stationary band-power node features:

    raw stream -> 0.5 s chunks -> PSD (Welch) -> theta/alpha/beta/gamma
    log-power + relative power -> (n_ch, 8) node features @ 2 Hz

Four consecutive chunks (~2 s) aggregate into one GNN frame, so the
spatial GNN always sees the same 2 s context the base model trained on,
while downstream consumers get fresh features every 0.5 s.

Filtering is streaming: notch + bandpass run as single-pass SOS filters
whose delay state carries across chunks (no edge artifacts, no filtfilt
warm-up), followed by a per-sample CAR spatial filter.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, welch

from gnn.model import CHANNELS, FS

# Standard bands: theta (drowsy/focus), alpha (relaxed/eyes-closed),
# beta (motor imagery, ERD on C3/C4), gamma (high-frequency activity).
FAST_BANDS = {"theta": (4, 8), "alpha": (8, 13),
              "beta": (13, 30), "gamma": (30, 40)}
BAND_ORDER = ("theta", "alpha", "beta", "gamma")
N_FAST_FEATURES = len(BAND_ORDER) * 2  # log-power + relative power per band

CHUNK_S = 0.5      # one PSD chunk: 125 samples @ 250 Hz
FRAME_CHUNKS = 4   # chunks per GNN frame (~2 s of context)


def welch_psd(epoch, fs=FS):
    """(C, T) -> (freqs, psd (C, F)); nperseg covers short 0.5 s chunks."""
    epoch = np.asarray(epoch, dtype=np.float64)
    nperseg = min(max(32, epoch.shape[-1]), 256)
    f, psd = welch(epoch, fs, nperseg=nperseg, axis=-1)
    return np.asarray(f), np.asarray(psd)


def bandpowers(epoch, fs=FS, bands=None):
    """(C, T) -> (C, B) mean absolute power per band (uV^2/Hz)."""
    bands = bands or BAND_ORDER
    f, psd = welch_psd(epoch, fs=fs)
    out = np.zeros((psd.shape[0], len(bands)))
    for j, name in enumerate(bands):
        lo, hi = FAST_BANDS[name]
        idx = (f >= lo) & (f < hi)
        if idx.any():
            out[:, j] = psd[:, idx].mean(axis=1)
    return out


def extract_chunk_features(chunk, fs=FS, mains=50):
    """One 0.5 s raw chunk (C, T) -> (C, 8) node features.

    Per band: log10 absolute power + relative power (band / broadband).
    Standardized across nodes, matching ``gnn.model`` conventions.
    """
    from gnn.model import _bandpass_sos, _notch_sos
    from scipy.signal import sosfiltfilt

    chunk = np.asarray(chunk, dtype=np.float64)
    x = sosfiltfilt(_notch_sos(mains, fs), chunk, axis=-1)
    x = sosfiltfilt(_bandpass_sos(fs), x, axis=-1)
    x = x - x.mean(axis=0, keepdims=True)  # CAR spatial filter
    bp = bandpowers(x, fs=fs)
    total = bp.sum(axis=1, keepdims=True) + 1e-12
    feats = np.concatenate([np.log10(bp + 1e-12), bp / total], axis=1)
    feats = feats.astype(np.float32)
    return (feats - feats.mean(0)) / (feats.std(0) + 1e-6)


class RollingPSD:
    """Streaming chunker: samples in, (C, 8) PSD features out every 0.5 s.

    Keeps filter state + the last ``FRAME_CHUNKS`` chunk features, so
    :meth:`frame` always returns the current ~2 s GNN context.
    """

    def __init__(self, n_ch=8, fs=FS, mains=50, chunk_s=CHUNK_S,
                 frame_chunks=FRAME_CHUNKS):
        from gnn.model import _bandpass_sos, _notch_sos

        self.n_ch, self.fs, self.mains = n_ch, fs, mains
        self.chunk_n = int(round(chunk_s * fs))
        self.frame_chunks = frame_chunks
        self._sos_nt = _notch_sos(mains, fs)
        self._sos_bp = _bandpass_sos(fs)
        n_sec = self._sos_bp.shape[0]
        # sosfilt zi layout for (C, T) input: (n_sections, C, 2).
        self._zi_nt = np.repeat(sosfilt_zi(self._sos_nt)[:, None, :],
                                n_ch, axis=1)
        self._zi_bp = np.repeat(sosfilt_zi(self._sos_bp)[:, None, :],
                                n_ch, axis=1)
        assert self._zi_bp.shape == (n_sec, n_ch, 2)
        self._pending = np.zeros((n_ch, 0))
        self._chunks = []  # recent (C, F) chunk features, oldest first
        self._series = np.zeros((n_ch, 0))  # filtered samples for connectivity

    @property
    def n_ready(self):
        return len(self._chunks)

    @property
    def frame_ready(self):
        return len(self._chunks) >= self.frame_chunks

    def push(self, samples):
        """Feed raw (C, T) samples; returns new chunk features list."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.shape[0] != self.n_ch:
            raise ValueError(f"expected {self.n_ch} channels, "
                             f"got {samples.shape[0]}")
        fresh, fz = sosfilt(self._sos_nt, samples, axis=-1, zi=self._zi_nt)
        self._zi_nt = fz
        fresh, fz = sosfilt(self._sos_bp, fresh, axis=-1, zi=self._zi_bp)
        self._zi_bp = fz
        fresh = fresh - fresh.mean(axis=0, keepdims=True)  # streaming CAR
        self._series = np.concatenate([self._series, fresh], axis=1)
        self._series = self._series[:, -self.chunk_n * self.frame_chunks:]
        self._pending = np.concatenate([self._pending, fresh], axis=1)
        new = []
        while self._pending.shape[1] >= self.chunk_n:
            piece, self._pending = (self._pending[:, :self.chunk_n],
                                    self._pending[:, self.chunk_n:])
            bp = bandpowers(piece, fs=self.fs)
            total = bp.sum(axis=1, keepdims=True) + 1e-12
            feats = np.concatenate([np.log10(bp + 1e-12), bp / total],
                                   axis=1).astype(np.float32)
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
            self._chunks.append(feats)
            self._chunks = self._chunks[-self.frame_chunks:]
            new.append(feats)
        return new

    def frame(self):
        """Mean of the buffered chunks -> (C, F) ~2 s GNN frame (or None)."""
        if not self.frame_ready:
            return None
        return np.mean(np.stack(self._chunks), axis=0).astype(np.float32)

    def frame_epoch(self):
        """Last ~2 s of filtered series for functional connectivity (or None)."""
        want = self.chunk_n * self.frame_chunks
        if not self.frame_ready or self._series.shape[1] < want:
            return None
        return np.ascontiguousarray(self._series[:, -want:])

    def latest_bands(self):
        """Last chunk's raw band powers, reshaped for summaries (or None)."""
        if not self._chunks:
            return None
        return self._chunks[-1]


class BrainAccessStream:
    """Iterate 0.5 s raw chunks from a BrainAccess Mini via the SDK.

    Thin wrapper over :class:`gnn.source.LiveSource` yielding fixed-size
    chunks (drops the first partial window while the headset buffer fills).
    Needs Windows + ``mne`` + the BrainAccess SDK + hardware; use
    :class:`gnn.source.ReplaySource` + :class:`RollingPSD` without hardware.
    """

    def __init__(self, device_name, chunk_s=CHUNK_S, sfreq=FS,
                 channels=None):
        self.device_name = device_name
        self.chunk_s = chunk_s
        self.sfreq = sfreq
        self.channels = list(channels or CHANNELS)
        self._source = None

    def __enter__(self):
        from gnn.source import LiveSource
        self._source = LiveSource(self.device_name, sfreq=self.sfreq)
        self._source.__enter__()
        return self

    def __exit__(self, *exc):
        if self._source is not None:
            self._source.__exit__(*exc)
            self._source = None

    def chunks(self, count=None):
        """Yield (C, chunk_n) raw chunks roughly every ``chunk_s``."""
        import time

        want = int(round(self.chunk_s * self.sfreq))
        n = 0
        while count is None or n < count:
            t0 = time.time()
            window = self._source.read_window(seconds=self.chunk_s,
                                              channels=self.channels)
            if window is not None and window.shape[1] >= want:
                yield np.ascontiguousarray(window[:, -want:])
                n += 1
            dt = time.time() - t0
            time.sleep(max(0.0, self.chunk_s - dt))
