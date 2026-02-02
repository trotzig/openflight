#!/usr/bin/env python3
"""
Sound trigger placement tester for SEN-14262 + OPS243-A.

Simulates the SEN-14262 hardware GATE signal using your computer's microphone.
When a loud sound is detected, sends S! to the radar (standing in for the
SEN-14262 GATE → J3 pin 3 HOST_INT rising edge). This lets you test the full
rolling buffer capture and processing flow before the sound sensor arrives.

Per AN-027 Figure 3: the SEN-14262 GATE connects directly to the OPS243-A
J3 pin 3 (HOST_INT). The radar detects the low→high edge and dumps its
rolling buffer. This script replaces that hardware path with:

    Mic → software peak detection → S! command → same buffer dump

The radar side is identical to production — real I/Q data, real FFT processing,
real speed extraction. Only the trigger source differs.

Usage:
    uv run python scripts/test_sound_trigger.py
    uv run python scripts/test_sound_trigger.py --threshold 0.3 --pre-trigger 16
    uv run python scripts/test_sound_trigger.py --port /dev/ttyACM0

What to test:
  - Place your mic where you plan to mount the SEN-14262
  - Clap hands / hit a ball into a net near the radar
  - Watch for real speed data in the capture output
  - Tune --threshold until triggers are reliable without false positives
  - Tune --pre-trigger (S#n) to capture enough pre-impact data
  - The peak audio level helps calibrate the SEN-14262 onboard pot later
"""

import argparse
import math
import struct
import sys
import time

try:
    import pyaudio
except ImportError:
    print("pyaudio is required for this script.")
    print("Install with: uv pip install pyaudio")
    print()
    print("On macOS: brew install portaudio")
    print("On Raspberry Pi / Debian: sudo apt install python3-pyaudio portaudio19-dev")
    sys.exit(1)

# Add src to path so we can import openflight
sys.path.insert(0, "src")

from openflight.ops243 import OPS243Radar  # noqa: E402
from openflight.rolling_buffer.processor import RollingBufferProcessor  # noqa: E402

# Audio settings
RATE = 44100
CHUNK = 1024  # ~23ms per chunk at 44.1kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
MAX_INT16 = 32768.0

# Radar constants (AN-027)
SAMPLES_PER_SEGMENT = 128
TOTAL_SEGMENTS = 32
SEGMENT_DURATION_MS = SAMPLES_PER_SEGMENT / 30000 * 1000  # ~4.27ms at 30ksps


def peak_level(data: bytes) -> float:
    """Calculate peak amplitude normalized to 0.0-1.0."""
    count = len(data) // 2
    samples = struct.unpack(f"<{count}h", data)
    return max(abs(s) for s in samples) / MAX_INT16


def rms_level(data: bytes) -> float:
    """Calculate RMS amplitude normalized to 0.0-1.0."""
    count = len(data) // 2
    samples = struct.unpack(f"<{count}h", data)
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / count) / MAX_INT16


def level_bar(level: float, width: int = 40) -> str:
    """Render a visual level meter."""
    filled = min(int(level * width), width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def process_capture(radar: OPS243Radar, processor: RollingBufferProcessor,
                    pre_trigger_segments: int, trigger_time: float, audio_peak: float,
                    trigger_count: int, swings_only: bool = False):
    """
    Send S! to the radar (standing in for SEN-14262 GATE), read the buffer
    dump, and process it through the real FFT pipeline.
    """
    pre_ms = pre_trigger_segments * SEGMENT_DURATION_MS
    post_ms = (TOTAL_SEGMENTS - pre_trigger_segments) * SEGMENT_DURATION_MS

    print()
    print("=" * 60)
    print(f"  TRIGGER #{trigger_count}")
    print("=" * 60)
    print(f"  Audio peak:     {audio_peak:.3f} ({audio_peak * 100:.1f}%)")
    print(f"  Capture window: {pre_ms:.1f}ms pre / {post_ms:.1f}ms post (S#{pre_trigger_segments})")
    print()

    # Send S! — this is what the SEN-14262 GATE→HOST_INT would do in hardware
    print("  Sending S! (software trigger, standing in for GATE→HOST_INT)...")
    s_bang_time = time.time()
    response = radar.trigger_capture(timeout=10.0)
    capture_done_time = time.time()

    transfer_ms = (capture_done_time - s_bang_time) * 1000
    print(f"  Buffer received: {len(response)} bytes in {transfer_ms:.0f}ms")

    # Re-arm for next capture (same as production: GC + PA)
    radar.rearm_rolling_buffer()

    # Parse the I/Q response
    capture = processor.parse_capture(response)

    if capture is None:
        print("  WARNING: Failed to parse I/Q data from buffer dump")
        print("=" * 60)
        print()
        return

    print(f"  I/Q samples:    {len(capture.i_samples)} I, {len(capture.q_samples)} Q")

    # Run real FFT processing — same pipeline as production
    timeline = processor.process_standard(capture)

    print()
    print(f"  Processed {len(timeline.readings)} speed readings across 32 segments:")
    print()

    # Show readings grouped by segment like AN-027 Appendix A output
    outbound_readings = []
    for reading in timeline.readings:
        direction = "OUT" if reading.is_outbound else " IN"
        marker = " <--" if reading.is_outbound and reading.speed_mph >= 15.0 else ""
        print(f"    Seg {reading.segment_index:>2}: {reading.speed_mph:>6.1f} mph "
              f"{direction} (mag: {reading.magnitude:>6.1f}){marker}")
        if reading.is_outbound and reading.speed_mph >= 15.0:
            outbound_readings.append(reading)

    print()
    if not outbound_readings:
        if swings_only:
            print("  No swing detected — discarding and re-arming.")
        else:
            print("  No swing detected (false trigger / nearby player)")
        print("=" * 60)
        print()
        return

    peak_reading = max(outbound_readings, key=lambda r: r.speed_mph)
    print(f"  Peak outbound:  {peak_reading.speed_mph:.1f} mph "
          f"(segment {peak_reading.segment_index}, mag {peak_reading.magnitude:.1f})")
    print(f"  Outbound count: {len(outbound_readings)} readings >= 15 mph")

    # Show timing summary relative to what hardware trigger would look like
    print()
    print("  Timing comparison (S! software vs SEN-14262 hardware):")
    print(f"    Software (this test):  ~{transfer_ms:.0f}ms from trigger to data")
    print("    Hardware (production): <1ms GATE latency + same serial transfer")
    print("=" * 60)
    print()


def run(port: str, threshold: float, pre_trigger_segments: int, cooldown: float,
        swings_only: bool = False):
    """Connect to radar, configure rolling buffer, then listen on mic."""

    # --- Connect to radar ---
    print("Connecting to radar...")
    radar = OPS243Radar(port=port if port else None)
    radar.connect()
    print(f"Connected on {radar.port}")

    # Get firmware info
    info = radar.get_info()
    version = info.get("Version", "unknown")
    print(f"Firmware: {version}")

    # --- Configure rolling buffer mode (same as production) ---
    print(f"Configuring rolling buffer mode (S#{pre_trigger_segments})...")
    radar.configure_for_rolling_buffer()
    radar.set_trigger_split(pre_trigger_segments)
    print("Radar ready — rolling buffer active, waiting for trigger")

    # --- Set up FFT processor ---
    processor = RollingBufferProcessor()

    # --- Open microphone ---
    pa = pyaudio.PyAudio()
    default_mic = pa.get_default_input_device_info()
    print()
    print(f"Mic: {default_mic['name']}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Pre-trigger: S#{pre_trigger_segments} "
          f"({pre_trigger_segments * SEGMENT_DURATION_MS:.1f}ms pre, "
          f"{(TOTAL_SEGMENTS - pre_trigger_segments) * SEGMENT_DURATION_MS:.1f}ms post)")
    print(f"Cooldown: {cooldown:.1f}s between triggers")
    print(f"Mode: {'swings only (ignoring false triggers)' if swings_only else 'all captures (debug)'}")
    print()
    print("Listening... clap or make a loud sound to trigger. Ctrl+C to quit.")
    print()

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    trigger_count = 0
    last_trigger_time = 0

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = rms_level(data)
            pk = peak_level(data)

            # Live level meter
            bar = level_bar(rms)
            sys.stdout.write(f"\r  {bar} RMS={rms:.3f} Peak={pk:.3f}")
            sys.stdout.flush()

            # Check for trigger
            now = time.time()
            if pk >= threshold and (now - last_trigger_time) > cooldown:
                trigger_count += 1
                last_trigger_time = now

                # Clear the level meter line
                sys.stdout.write("\r" + " " * 80 + "\r")

                process_capture(
                    radar, processor, pre_trigger_segments,
                    now, pk, trigger_count, swings_only,
                )

                print("Listening...")
                print()

    except KeyboardInterrupt:
        print()
        print()
        print(f"Done. {trigger_count} triggers fired.")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Disconnecting radar...")
        radar.disable_rolling_buffer()
        radar.disconnect()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Test sound trigger placement using real OPS243-A radar in rolling buffer mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How it works:
  Per AN-027 Figure 3, the SEN-14262 GATE output connects to OPS243-A J3
  pin 3 (HOST_INT). A loud sound causes a low→high edge that triggers the
  rolling buffer dump. This script replaces that hardware GATE with a
  software S! command fired when your mic detects a loud sound.

  The radar side is identical to production: real rolling buffer, real I/Q
  data, real FFT processing. Only the trigger source differs.

Examples:
  # Connect to radar (auto-detect port), default threshold
  uv run python scripts/test_sound_trigger.py

  # Specify port explicitly
  uv run python scripts/test_sound_trigger.py --port /dev/ttyACM0

  # More sensitive threshold (picks up softer sounds)
  uv run python scripts/test_sound_trigger.py --threshold 0.15

  # More pre-trigger history (16 segments = ~68ms before impact)
  uv run python scripts/test_sound_trigger.py --pre-trigger 16

Tips:
  - Place your mic where you plan to mount the SEN-14262
  - Start with default threshold, clap at that distance
  - If false triggers, increase --threshold
  - If claps don't trigger, decrease --threshold
  - The audio peak level helps you calibrate the SEN-14262 pot later
  - Wave your hand in front of the radar while triggering to see real speeds
        """,
    )
    parser.add_argument(
        "--port", help="Serial port for radar (auto-detect if not specified)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="Audio trigger threshold 0.0-1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--pre-trigger", "-p", type=int, default=12,
        help="Pre-trigger segments S#n, 0-32 (default: 12, ~51ms pre-trigger)"
    )
    parser.add_argument(
        "--cooldown", "-c", type=float, default=3.0,
        help="Seconds between triggers (default: 3.0, accounts for serial transfer)"
    )
    parser.add_argument(
        "--swings-only", "-s", action="store_true",
        help="Only show captures with real swing data (>= 15 mph outbound). "
             "Without this flag, all captures are shown for debugging."
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        print("Error: threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if not 0 <= args.pre_trigger <= 32:
        print("Error: pre-trigger must be between 0 and 32")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  OpenFlight Sound Trigger Placement Tester")
    print("  (mic → S! standing in for SEN-14262 GATE → HOST_INT)")
    print("=" * 60)
    print()

    run(args.port, args.threshold, args.pre_trigger, args.cooldown, args.swings_only)


if __name__ == "__main__":
    main()
