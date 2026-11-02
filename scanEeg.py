import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for Bluetooth devices (make sure your BrainAccess Mini is on)...")
    devices = await BleakScanner.discover()
    for d in devices:
        # Look for devices that might match BrainAccess or show their names/addresses
        print(f"Device: {d.name} | Address: {d.address}")

asyncio.run(main())