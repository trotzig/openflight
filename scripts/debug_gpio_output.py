#!/usr/bin/env python3
"""
Simple GPIO output diagnostic - toggles GPIO27 slowly for multimeter verification.

This helps isolate whether the issue is:
1. GPIO not outputting voltage
2. Wiring between GPIO27 and J3 Pin 3
3. J3 pin identification

Usage:
    uv run python scripts/debug_gpio_output.py
    uv run python scripts/debug_gpio_output.py --max-drive   # Use 16mA drive strength

While running, measure with multimeter:
    - Between GPIO27 (physical pin 13) and GND (physical pin 14)
    - Should see 0V → 3.3V → 0V cycling every 2 seconds
"""

import argparse
import subprocess
import sys
import time

try:
    import lgpio
except ImportError:
    print("ERROR: lgpio not available")
    print("Install with: pip install lgpio")
    sys.exit(1)

OUTPUT_PIN = 27  # BCM 27 = Physical pin 13


def set_pad_drive_strength(strength_ma: int = 16):
    """
    Set GPIO pad drive strength using raspi-gpio or pinctrl.

    GPIO27 is on pad group 0 (GPIOs 0-27).
    Valid strengths: 2, 4, 6, 8, 10, 12, 14, 16 mA
    """
    print(f"Setting GPIO pad drive strength to {strength_ma}mA...")

    # Try raspi-gpio first (older Pi OS)
    try:
        result = subprocess.run(
            ["raspi-gpio", "pads", "0", str(strength_ma)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  Set via raspi-gpio: pad 0 = {strength_ma}mA")
            return True
        else:
            print(f"  raspi-gpio failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  raspi-gpio not found, trying pinctrl...")

    # Try pinctrl (newer Pi OS / Pi 5)
    try:
        result = subprocess.run(
            ["pinctrl", "set", str(OUTPUT_PIN), f"op", f"dl"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  pinctrl configured GPIO{OUTPUT_PIN} as output")
    except FileNotFoundError:
        print("  pinctrl not found")

    # Suggest manual command
    print()
    print("  To manually set max drive strength, run:")
    print(f"    sudo raspi-gpio pads 0 {strength_ma}")
    print("  Or on Pi 5:")
    print(f"    sudo pinctrl set {OUTPUT_PIN} op")
    print()
    return False


def main():
    parser = argparse.ArgumentParser(description="GPIO output diagnostic")
    parser.add_argument(
        "--max-drive", "-m", action="store_true",
        help="Set maximum GPIO drive strength (16mA)"
    )
    parser.add_argument(
        "--drive-strength", "-d", type=int, default=16,
        choices=[2, 4, 6, 8, 10, 12, 14, 16],
        help="GPIO drive strength in mA (default: 16)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  GPIO Output Diagnostic")
    print("=" * 70)
    print()

    if args.max_drive:
        set_pad_drive_strength(args.drive_strength)
        print()

    print("This script toggles GPIO27 HIGH/LOW every 2 seconds.")
    print("Use a multimeter to verify voltage output.")
    print()
    print("MEASUREMENT POINTS:")
    print("  Positive probe: Physical pin 13 (GPIO27)")
    print("  Negative probe: Physical pin 14 (GND)")
    print()
    print("EXPECTED READINGS:")
    print("  When script says 'HIGH': ~3.3V")
    print("  When script says 'LOW':  ~0V")
    print()
    print("Raspberry Pi GPIO Header (looking at Pi with GPIO header on right):")
    print()
    print("         3V3  (1)  (2)  5V")
    print("       GPIO2  (3)  (4)  5V")
    print("       GPIO3  (5)  (6)  GND")
    print("       GPIO4  (7)  (8)  GPIO14")
    print("         GND  (9) (10)  GPIO15")
    print("      GPIO17 (11) (12)  GPIO18")
    print("  >>> GPIO27 (13) (14)  GND <<<  <- Measure between these two")
    print("      GPIO22 (15) (16)  GPIO23")
    print("         3V3 (17) (18)  GPIO24")
    print("      GPIO10 (19) (20)  GND")
    print()
    print("-" * 70)
    print()

    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, OUTPUT_PIN, 0)

    print("Starting toggle test. Press Ctrl+C to stop.")
    print()

    try:
        cycle = 0
        while True:
            cycle += 1

            # HIGH
            lgpio.gpio_write(h, OUTPUT_PIN, 1)
            print(f"[{cycle}] GPIO27 = HIGH (should read ~3.3V)")
            time.sleep(2.0)

            # LOW
            lgpio.gpio_write(h, OUTPUT_PIN, 0)
            print(f"[{cycle}] GPIO27 = LOW  (should read ~0V)")
            time.sleep(2.0)

    except KeyboardInterrupt:
        print()
        print("Stopped.")

    lgpio.gpio_write(h, OUTPUT_PIN, 0)
    lgpio.gpiochip_close(h)

    print()
    print("NEXT STEPS:")
    print()
    print("If multimeter shows correct 0V/3.3V cycling:")
    print("  → GPIO output is working. Issue is wiring to radar.")
    print("  → Verify wire goes from Pi pin 13 to J3 Pin 3 (HOST_INT)")
    print("  → Verify Pi GND (pin 14) connects to J3 Pin 10 (GND)")
    print()
    print("If multimeter shows 0V always or wrong voltage:")
    print("  → GPIO not working. Check lgpio installation.")
    print("  → Try: sudo apt install python3-lgpio")
    print()
    print("J3 HEADER ON OPS243-A (10-pin, looking at component side):")
    print()
    print("  Pin 1 (GPIO)      Pin 2 (GPIO)")
    print("  Pin 3 (HOST_INT)  Pin 4 (/RESET)")
    print("  Pin 5 (SPI_SEL)   Pin 6 (RxD)")
    print("  Pin 7 (TxD)       Pin 8 (SCK)")
    print("  Pin 9 (5V)        Pin 10 (GND)")
    print()
    print("Wire connections needed:")
    print("  Pi GPIO27 (pin 13) ──────► J3 Pin 3 (HOST_INT)")
    print("  Pi GND (pin 14)    ──────► J3 Pin 10 (GND)")
    print()


if __name__ == "__main__":
    main()
