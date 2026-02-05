#!/usr/bin/env python3
"""
Quick debug script to test if the hardware trigger is working.
"""

import sys
sys.path.insert(0, "src")

from openflight.ops243 import OPS243Radar

print("Connecting to radar...")
radar = OPS243Radar()
radar.connect()
print(f"Connected on {radar.port}")

print("Configuring rolling buffer mode...")
radar.configure_for_rolling_buffer()
radar.set_trigger_split(12)

print()
print("Waiting 30s for hardware trigger - make a loud sound...")
print()

response = radar.wait_for_hardware_trigger(timeout=30)

if response:
    print(f"SUCCESS! Got {len(response)} bytes")
    print()
    print("First 500 chars:")
    print(response[:500])
else:
    print("No trigger received within 30s")

print()
print("Disconnecting...")
radar.disconnect()
print("Done.")
