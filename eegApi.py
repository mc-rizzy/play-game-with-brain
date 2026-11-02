import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "1A0D1730-F614-2580-2A99-0529A1D3E3DD"
CONTROL_AND_DATA_CHAR = "c71a59d2-d06c-42ce-a23b-30fca36a11fc"

received_packets = 0

def notification_handler(sender, data):
    global received_packets
    received_packets += 1
    # Print length and sample of data to see if it's EEG payload or an ACK code
    print(f"📥 Packet #{received_packets} from {sender} ({len(data)} bytes): {data.hex()}")

async def run():
    print(f"Connecting to BrainAccess Mini at {DEVICE_ADDRESS}...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print("Connected successfully!\n")
            
            # Subscribe to notifications on the control/data characteristic
            await client.start_notify(CONTROL_AND_DATA_CHAR, notification_handler)
            print(f"Subscribed to: {CONTROL_AND_DATA_CHAR}")

            # Let's try a standard multi-byte command packet or a different start command
            # Some devices expect [0x02, 0x01] or a start stream header
            commands_to_try = [
                b'\x01\x01',
                b'\x02\x01',
                b'\x00',
                b'START'
            ]

            for cmd in commands_to_try:
                print(f"\nSending command payload: {cmd}")
                await client.write_gatt_char(CONTROL_AND_DATA_CHAR, cmd)
                await asyncio.sleep(2) # Wait a moment to see if it triggers streaming
                
                if received_packets > 1:
                    print("Stream successfully triggered!")
                    break

            if received_packets <= 1:
                print("\nNo stream triggered yet, keeping connection open for 15 seconds to listen manually...")
                await asyncio.sleep(15)

            print("\nDisconnecting sequence initiated.")

asyncio.run(run())