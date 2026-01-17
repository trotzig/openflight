#!/usr/bin/env python3
"""
Capture raw I/Q data from rolling buffer and save to pickle file.
This allows offline analysis and debugging of signal processing.

Usage:
    python scripts/capture_iq.py [output_file.pkl]

Default output: ~/openflight_sessions/iq_captures_YYYYMMDD_HHMMSS.pkl
"""

import pickle
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from openflight.ops243 import OPS243Radar


def parse_capture(response: str) -> dict | None:
    """Parse S! command response into raw data dict."""
    sample_time = None
    trigger_time = None
    i_samples = None
    q_samples = None

    for line in response.strip().split('\n'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            data = json.loads(line)
            if "sample_time" in data:
                sample_time = float(data["sample_time"])
            elif "trigger_time" in data:
                trigger_time = float(data["trigger_time"])
            elif "I" in data:
                i_samples = np.array(data["I"], dtype=np.int16)
            elif "Q" in data:
                q_samples = np.array(data["Q"], dtype=np.int16)
        except json.JSONDecodeError:
            continue

    if i_samples is None or q_samples is None:
        return None

    # Compute derived values (same as processor.py)
    i_centered = i_samples.astype(np.float64) - np.mean(i_samples)
    q_centered = q_samples.astype(np.float64) - np.mean(q_samples)

    # Scale to voltage
    i_scaled = i_centered * (3.3 / 4096)
    q_scaled = q_centered * (3.3 / 4096)

    # Complex signal (standard I + jQ)
    complex_signal = i_scaled + 1j * q_scaled

    return {
        "sample_time": sample_time,
        "trigger_time": trigger_time,
        "i_raw": i_samples,
        "q_raw": q_samples,
        "i_centered": i_centered,
        "q_centered": q_centered,
        "i_scaled": i_scaled,
        "q_scaled": q_scaled,
        "complex_signal": complex_signal,
        "capture_timestamp": datetime.now().isoformat(),
        "response_length": len(response),
    }


def main():
    # Output file - save to ~/openflight_sessions by default
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        output_dir = Path.home() / "openflight_sessions"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"iq_captures_{timestamp}.pkl"

    print(f"=== I/Q Capture Tool ===")
    print(f"Output file: {output_path}")
    print("Connecting to radar...")

    radar = OPS243Radar()
    radar.connect()

    # Get radar info
    info = radar.get_info()
    print(f"Radar: {info.get('Product', 'unknown')} v{info.get('Version', 'unknown')}")
    print(f"Sample rate: {info.get('SamplingRate', 'unknown')} Hz")

    # Configure for rolling buffer
    print("Configuring rolling buffer mode...")
    radar.configure_for_rolling_buffer()

    captures = []
    metadata = {
        "radar_info": info,
        "capture_start": datetime.now().isoformat(),
        "sample_rate": 30000,
        "fft_size": 4096,
        "window_size": 128,
    }

    print("\nCapturing I/Q data (including noise/baseline).")
    print("Swing a club or hit balls to capture motion data.")
    print("Press Ctrl+C to stop and save\n")

    capture_count = 0
    try:
        while True:
            # Trigger capture (same as diagnose_fft.py)
            response = radar.trigger_capture(timeout=10.0)

            # Re-arm for next capture
            radar.rearm_rolling_buffer()

            # Try to parse - capture ALL valid responses
            capture = parse_capture(response)

            if capture:
                capture_count += 1
                captures.append(capture)

                # Show quick stats
                i_std = np.std(capture["i_raw"])
                q_std = np.std(capture["q_raw"])
                i_range = capture['i_raw'].max() - capture['i_raw'].min()
                q_range = capture['q_raw'].max() - capture['q_raw'].min()

                # Indicate activity level
                if i_range > 1000 or q_range > 1000:
                    activity = "HIGH"
                elif i_range > 200 or q_range > 200:
                    activity = "med"
                else:
                    activity = "low"

                print(f"[{capture_count}] {activity}: I({capture['i_raw'].min()}-{capture['i_raw'].max()}, std={i_std:.0f}) "
                      f"Q({capture['q_raw'].min()}-{capture['q_raw'].max()}, std={q_std:.0f})")
            else:
                # Failed to parse
                print(f"x", end="", flush=True)

            time.sleep(0.3)

    except KeyboardInterrupt:
        print(f"\n\nStopping... captured {capture_count} frames")
    finally:
        radar.disconnect()

    # Save to pickle
    if captures:
        metadata["capture_end"] = datetime.now().isoformat()
        output_data = {
            "metadata": metadata,
            "captures": captures,
        }

        with open(output_path, "wb") as f:
            pickle.dump(output_data, f)

        print(f"\nSaved {len(captures)} captures to {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

        print("\nTo analyze in Python:")
        print(f"  import pickle")
        print(f"  with open('{output_path}', 'rb') as f:")
        print(f"      data = pickle.load(f)")
        print(f"  captures = data['captures']")
        print(f"  # Each capture has: i_raw, q_raw, complex_signal, etc.")
    else:
        print("\nNo captures to save.")


if __name__ == "__main__":
    main()
