import time
from brainaccess.utils.acquisition import EEG

# 1. Initialize the EEG acquisition manager
eeg = EEG(mode="accumulate")  # 'accumulate' gathers continuous raw data

try:
    print("Scanning and setting up BrainAccess device...")
    # Scans and connects to the available headset
    eeg.setup()

    print("Starting raw EEG acquisition stream...")
    eeg.start_acquisition()

    # Stream data for 5 seconds
    duration = 5
    for elapsed in range(1, duration + 1):
        time.sleep(1)
        print(f"Streaming raw data... {elapsed}/{duration}s")

    # Stop data acquisition
    eeg.stop_acquisition()
    print("Stream stopped.")

    # 2. Retrieve raw data as an MNE Raw object
    mne_raw = eeg.get_mne()

    # 3. Extract raw numpy array data and channel names
    raw_data, times = mne_raw.get_data(return_times=True)
    ch_names = mne_raw.info['ch_names']
    sfreq = mne_raw.info['sfreq']

    print("\n--- RAW DATA RESULTS ---")
    print(f"Sampling Rate: {sfreq} Hz")
    print(f"Channels ({len(ch_names)}): {ch_names}")
    print(f"Raw Matrix Shape (Channels x Samples): {raw_data.shape}")
    print("\nSample of First 5 Signal Values from Channel 0:")
    print(raw_data[0, :5])

finally:
    # Always close the session to release hardware locks
    eeg.close()