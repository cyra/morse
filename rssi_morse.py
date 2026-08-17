#!/usr/bin/env python3
"""WiFi RSSI Hand-Gesture Morse Code Detector.

Uses WiFi signal strength (RSSI) to detect hand blockages
in the signal path and decode them as Morse code.
No Nexmon CSI required — works with stock Pi WiFi.
"""

import subprocess
import time
import sys
import re
from collections import deque

# --- Morse code dictionary ---
MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9", "-----": "0",
}

# --- Timing thresholds (seconds) ---
DOT_MAX = 0.3
DASH_MAX = 0.8
LETTER_GAP = 0.8
WORD_GAP = 2.0

# --- Detection parameters ---
RSSI_DROP_DB = 8         # dB drop from baseline to count as blocked
EMA_ALPHA = 0.02         # baseline adaptation speed
MIN_BLOCKAGE_DURATION = 0.08
POLL_INTERVAL = 0.02     # 20ms polling (~50 Hz)
INTERFACE = "wlan0"

# --- Calibration ---
CALIBRATION_SAMPLES = 50


def get_rssi() -> float | None:
    """Read current RSSI from /proc/net/wireless (fastest method)."""
    try:
        with open("/proc/net/wireless") as f:
            for line in f:
                if INTERFACE in line:
                    parts = line.split()
                    # Format: iface status link level noise ...
                    # level is field index 3, may have trailing '.'
                    return float(parts[3].rstrip("."))
    except (IOError, IndexError, ValueError):
        pass
    return None


def decode_morse(symbols: str) -> str:
    return MORSE_TO_CHAR.get(symbols, "?")


def main():
    print("=" * 50)
    print("WiFi RSSI Morse Code Detector")
    print("=" * 50)
    print(f"Interface: {INTERFACE}")
    print(f"Polling at {1/POLL_INTERVAL:.0f} Hz")
    print(f"Drop threshold: {RSSI_DROP_DB} dB below baseline")
    print(f"Dot: <{DOT_MAX}s  Dash: {DOT_MAX}-{DASH_MAX}s")
    print()

    # Quick check
    rssi = get_rssi()
    if rssi is None:
        print(f"ERROR: Cannot read RSSI from {INTERFACE}.")
        print("Make sure you're connected to a WiFi network.")
        sys.exit(1)
    print(f"Current RSSI: {rssi} dBm")

    # Calibrate
    print(f"Calibrating ({CALIBRATION_SAMPLES} samples)... keep hands clear.")
    samples = []
    for _ in range(CALIBRATION_SAMPLES):
        r = get_rssi()
        if r is not None:
            samples.append(r)
        time.sleep(POLL_INTERVAL)

    if len(samples) < 10:
        print("ERROR: Not enough RSSI samples during calibration.")
        sys.exit(1)

    baseline = sum(samples) / len(samples)
    noise_floor = max(abs(s - baseline) for s in samples)
    print(f"Baseline: {baseline:.1f} dBm (noise: +/-{noise_floor:.1f} dB)")

    # Auto-adjust threshold if noise is high
    effective_drop = max(RSSI_DROP_DB, noise_floor * 2 + 1)
    if effective_drop != RSSI_DROP_DB:
        print(f"Adjusted drop threshold to {effective_drop:.1f} dB (noisy environment)")
    else:
        effective_drop = RSSI_DROP_DB

    print()
    print("Ready! Block the signal path between Pi and router to input Morse code.")
    print("Dots/dashes shown on stderr, decoded letters on stdout.")
    print("-" * 50)

    blocked = False
    block_start = 0.0
    last_unblock_time = time.time()
    symbol_buffer = ""
    history = deque(maxlen=5)  # smooth over last 5 readings

    try:
        while True:
            rssi = get_rssi()
            if rssi is None:
                time.sleep(POLL_INTERVAL)
                continue

            history.append(rssi)
            smoothed = sum(history) / len(history)
            now = time.time()

            threshold = baseline - effective_drop

            if smoothed < threshold and not blocked:
                blocked = True
                block_start = now

            elif smoothed >= threshold and blocked:
                blocked = False
                duration = now - block_start
                last_unblock_time = now

                if duration < MIN_BLOCKAGE_DURATION:
                    pass
                elif duration < DOT_MAX:
                    symbol_buffer += "."
                    sys.stderr.write(".")
                    sys.stderr.flush()
                elif duration < DASH_MAX:
                    symbol_buffer += "-"
                    sys.stderr.write("-")
                    sys.stderr.flush()
                else:
                    if symbol_buffer:
                        letter = decode_morse(symbol_buffer)
                        print(letter, end="", flush=True)
                        symbol_buffer = ""
                    sys.stderr.write("|")
                    sys.stderr.flush()

            # Update baseline when not blocked (slow adaptation)
            if not blocked:
                baseline = (1 - EMA_ALPHA) * baseline + EMA_ALPHA * smoothed

            # Letter gap timeout
            if not blocked and symbol_buffer:
                gap = now - last_unblock_time
                if gap > LETTER_GAP:
                    letter = decode_morse(symbol_buffer)
                    print(letter, end="", flush=True)
                    symbol_buffer = ""
                    sys.stderr.write(" ")
                    sys.stderr.flush()

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        if symbol_buffer:
            letter = decode_morse(symbol_buffer)
            print(letter, end="", flush=True)
        print()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
