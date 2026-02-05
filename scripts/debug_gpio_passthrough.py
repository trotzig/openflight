#!/usr/bin/env python3
"""
Debug script for GPIO passthrough trigger.

Uses the exact same radar setup as test_sound_trigger_gpio.py
"""

import sys
import time

sys.path.insert(0, "src")

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False
    print("WARNING: lgpio not available")

from openflight.ops243 import OPS243Radar
from openflight.rolling_buffer.processor import RollingBufferProcessor

OUTPUT_PIN = 27  # HOST_INT output


def gpio_to_physical(bcm_pin: int) -> int:
    bcm_to_physical = {
        17: 11, 27: 13, 22: 15, 5: 29, 6: 31, 13: 33,
    }
    return bcm_to_physical.get(bcm_pin, 0)


def main():
    print("=" * 70)
    print("  GPIO Passthrough HOST_INT Debugger")
    print("=" * 70)
    print()

    # =========================================================================
    # Connect to radar (same as test_sound_trigger_gpio.py)
    # =========================================================================
    print("Connecting to radar...")
    radar = OPS243Radar()
    radar.connect()
    print(f"  Connected on: {radar.port}")

    info = radar.get_info()
    print(f"  Firmware: {info.get('Version', 'unknown')}")
    print(f"  Product: {info.get('Product', 'unknown')}")

    # =========================================================================
    # Configure rolling buffer (same as test_sound_trigger_gpio.py)
    # =========================================================================
    print()
    print("Configuring rolling buffer mode...")
    radar.configure_for_rolling_buffer()
    radar.set_trigger_split(12)
    print("  Done.")

    # =========================================================================
    # Test software trigger S! (same as test_sound_trigger_gpio.py)
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 1: Software trigger (S!)")
    print("-" * 70)

    processor = RollingBufferProcessor()

    print("Calling radar.trigger_capture()...")
    start = time.perf_counter()
    response = radar.trigger_capture(timeout=10.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"  Response time: {elapsed_ms:.0f}ms")
    print(f"  Response length: {len(response) if response else 0} bytes")

    if response and len(response) > 100:
        # Try to parse
        capture = processor.parse_capture(response)
        if capture:
            print(f"  I samples: {len(capture.i_samples)}")
            print(f"  Q samples: {len(capture.q_samples)}")
            print("  SUCCESS: Software trigger works!")
            sw_trigger_works = True
        else:
            print(f"  Failed to parse. Response preview: {response[:200]}")
            sw_trigger_works = False
    else:
        print(f"  FAIL: Response too short or empty")
        print(f"  Response: {response}")
        sw_trigger_works = False

    if not sw_trigger_works:
        print()
        print("Software trigger not working. Cannot test hardware trigger.")
        radar.disconnect()
        return

    # Re-arm
    radar.rearm_rolling_buffer()
    time.sleep(0.2)

    # =========================================================================
    # Test hardware trigger via GPIO
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 2: Hardware trigger (GPIO pulse to HOST_INT)")
    print("-" * 70)

    if not LGPIO_AVAILABLE:
        print("SKIP: lgpio not available")
        radar.disconnect()
        return

    print(f"Setting up GPIO{OUTPUT_PIN} (physical pin {gpio_to_physical(OUTPUT_PIN)})...")
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, OUTPUT_PIN, 0)
    print("  GPIO configured as output, currently LOW")

    # Clear any pending serial data
    radar.serial.reset_input_buffer()

    # Test rising edge
    print()
    print("Sending rising edge pulse (LOW → HIGH for 10ms → LOW)...")
    start = time.perf_counter()

    lgpio.gpio_write(h, OUTPUT_PIN, 1)
    time.sleep(0.010)
    lgpio.gpio_write(h, OUTPUT_PIN, 0)

    print("  Pulse sent, waiting for response...")

    # Read response using same method as wait_for_hardware_trigger
    response = radar.wait_for_hardware_trigger(timeout=5.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"  Response time: {elapsed_ms:.0f}ms")
    print(f"  Response length: {len(response) if response else 0} bytes")

    if response and len(response) > 100:
        capture = processor.parse_capture(response)
        if capture:
            print(f"  I samples: {len(capture.i_samples)}")
            print(f"  Q samples: {len(capture.q_samples)}")
            print("  SUCCESS: Hardware trigger (rising edge) works!")
        else:
            print(f"  Response received but parse failed: {response[:200]}")
    else:
        print("  FAIL: No I/Q data from rising edge trigger")

        # Try falling edge
        print()
        print("Trying falling edge (HIGH → LOW for 10ms → HIGH)...")
        radar.rearm_rolling_buffer()
        time.sleep(0.2)
        radar.serial.reset_input_buffer()

        lgpio.gpio_write(h, OUTPUT_PIN, 1)
        time.sleep(0.1)

        start = time.perf_counter()
        lgpio.gpio_write(h, OUTPUT_PIN, 0)
        time.sleep(0.010)
        lgpio.gpio_write(h, OUTPUT_PIN, 1)

        response = radar.wait_for_hardware_trigger(timeout=5.0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"  Response time: {elapsed_ms:.0f}ms")
        print(f"  Response length: {len(response) if response else 0} bytes")

        if response and len(response) > 100:
            print("  SUCCESS: Hardware trigger (falling edge) works!")
        else:
            print("  FAIL: No I/Q data from falling edge trigger")
            print()
            print("  Troubleshooting:")
            print("    1. Verify GPIO27 is connected to J3 Pin 3 (HOST_INT)")
            print("    2. Add GND jumper: Pi GND → J3 Pin 1")
            print("    3. Measure voltage on HOST_INT during pulse")

    lgpio.gpiochip_close(h)
    radar.disconnect()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
