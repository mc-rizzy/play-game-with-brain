import time
from brainaccess.core.eeg_manager import EEGManager
from brainaccess.utils.acquisition import EEG

# Define your specific device name (change this to match your headset label)
device_name = "BA MINI 048"  # Or whatever your headset is named

# Initialize the EEG acquisition manager object
eeg = EEG(mode="accumulate")  # 'accumulate' gathers continuous raw data

try:
    print("Scanning and setting up BrainAccess device...")
    
    # Use the EEGManager context manager and pass it into eeg.setup()
    with EEGManager() as mgr:
        # Pass the manager instance and device name
        eeg.setup(mgr, device_name=device_name)

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

        # Retrieve raw data as an MNE Raw object
        mne_raw = eeg.get_mne()

        # Extract raw numpy array data and channel names
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