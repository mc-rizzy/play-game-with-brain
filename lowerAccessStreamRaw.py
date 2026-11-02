import time
from brainaccess.core import core
from brainaccess.core.eeg_manager import EEGManager

# Initialize core runtime environment
core.init()

mgr = EEGManager()

# Scan for connected Bluetooth/USB headsets
devices = mgr.get_devices()
print(f"Found devices: {devices}")

if devices:
    device = devices[0]
    mgr.connect(device)
    
    # Load default hardware config and start stream
    mgr.load_config()
    mgr.start_stream()
    print("Streaming raw data...")

    time.sleep(3)

    # Fetch raw data chunks from the buffer
    data = mgr.get_data()
    print(f"Received raw data chunk shape: {data.shape}")

    mgr.stop_stream()
    mgr.disconnect()

core.close()