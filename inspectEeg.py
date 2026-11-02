import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "1A0D1730-F614-2580-2A99-0529A1D3E3DD"
CONTROL_CHAR = "c71a59d2-d06c-42ce-a23b-30fca36a11fc"

def notification_handler(sender, data):
    print(f"🎉 [DATA PACKET] Received {len(data)} bytes: {data.hex()}")

async def run():
    print(f"Connecting to BrainAccess Mini...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print("Connected successfully!\n")
            
            # Subscribe to notifications
            await client.start_notify(CONTROL_CHAR, notification_handler)
            
            # Let's try sending a command packet structured with a standard header or byte array
            # Often it's b'b' (begin) or a packet like [0xbc, 0x01] or similar
            commands = [
                b'\xbc\x01',          # Common EEG start frame
                b'\xc0\x00',          # Stream command
                b'b',                 # 'b' for begin
                bytes([0x01, 0x00])   # Start flag
            ]

            for cmd in commands:
                print(f"Trying packet: {cmd}")
                try:
                    await client.write_gatt_char(CONTROL_CHAR, cmd, response=False)
                except Exception as e:
                    print(f"  Error: {e}")
                await asyncio.sleep(2)

            print("\nListening for data streams for 10 seconds...")
            await asyncio.sleep(10)
            print("Disconnecting.")

asyncio.run(run())