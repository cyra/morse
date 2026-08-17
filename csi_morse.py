#!/usr/bin/env python3
"""WiFi CSI Hand-Gesture Morse Code Detector.

Uses Nexmon CSI on a Raspberry Pi 4B to detect hand blockages
in the WiFi signal path and decode them as Morse code.
"""

import socket
import struct
import time
import sys
import numpy as np

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
DOT_MAX = 0.3       # blockage < 0.3s = dot
DASH_MAX = 0.8      # blockage 0.3s-0.8s = dash; >0.8s = letter boundary
LETTER_GAP = 0.8    # gap after signal returns to emit letter
WORD_GAP = 2.0      # gap to emit word separator (space)

# --- Detection parameters ---
AMPLITUDE_DROP_FACTOR = 0.7   # amplitude must drop below baseline * factor
EMA_ALPHA = 0.01              # exponential moving average smoothing for baseline
MIN_BLOCKAGE_DURATION = 0.05  # ignore blockages shorter than 50ms (noise)

# --- Nexmon CSI UDP config ---
CSI_UDP_PORT = 5500
CSI_UDP_IP = "0.0.0.0"

# --- Nexmon CSI header: 18 bytes ---
# See https://github.com/seemoo-lab/nexmon_csi
NEXMON_HEADER_SIZE = 18


def parse_csi_packet(data: bytes) -> np.ndarray | None:
    """Parse a Nexmon CSI UDP packet and return complex CSI values.

    The Nexmon CSI packet format:
    - 2 bytes: magic (0x1111)
    - 1 byte: RSSI (signed)
    - 1 byte: frame control
    - 6 bytes: source MAC
    - 2 bytes: sequence number
    - 2 bytes: core and spatial stream
    - 2 bytes: chanspec
    - 2 bytes: chip version
    After the 18-byte header, the rest is CSI data as int16 pairs (I/Q).
    """
    if len(data) < NEXMON_HEADER_SIZE + 4:
        return None

    # Check magic bytes
    magic = struct.unpack(">H", data[:2])[0]
    if magic != 0x1111:
        return None

    csi_bytes = data[NEXMON_HEADER_SIZE:]

    # CSI data is pairs of int16 (I, Q) in big-endian
    n_values = len(csi_bytes) // 4  # each complex value is 4 bytes (2x int16)
    if n_values == 0:
        return None

    csi_complex = np.zeros(n_values, dtype=np.complex64)
    for i in range(n_values):
        offset = i * 4
        real = struct.unpack(">h", csi_bytes[offset:offset + 2])[0]
        imag = struct.unpack(">h", csi_bytes[offset + 2:offset + 4])[0]
        csi_complex[i] = complex(real, imag)

    return csi_complex


def compute_amplitude(csi: np.ndarray) -> float:
    """Compute mean amplitude across all subcarriers."""
    return float(np.mean(np.abs(csi)))


def decode_morse(symbols: str) -> str:
    """Decode a Morse symbol string to a character."""
    return MORSE_TO_CHAR.get(symbols, "?")


def main():
    print("=" * 50)
    print("WiFi CSI Morse Code Detector")
    print("=" * 50)
    print(f"Listening for CSI on UDP port {CSI_UDP_PORT}...")
    print(f"Dot: <{DOT_MAX}s  Dash: {DOT_MAX}-{DASH_MAX}s  Letter gap: >{LETTER_GAP}s")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((CSI_UDP_IP, CSI_UDP_PORT))
    sock.settimeout(0.01)  # 10ms timeout for responsive loop

    baseline = None
    blocked = False
    block_start = 0.0
    last_unblock_time = time.time()
    symbol_buffer = ""
    packet_count = 0
    calibrating = True
    calibration_amplitudes = []
    CALIBRATION_PACKETS = 100

    print("Calibrating baseline... keep hands clear of the signal path.")

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                # Check for letter gap timeout even when no packets arrive
                if symbol_buffer and not blocked:
                    gap = time.time() - last_unblock_time
                    if gap > LETTER_GAP:
                        letter = decode_morse(symbol_buffer)
                        print(letter, end="", flush=True)
                        symbol_buffer = ""
                continue

            csi = parse_csi_packet(data)
            if csi is None:
                continue

            amplitude = compute_amplitude(csi)
            packet_count += 1

            # Calibration phase: collect initial amplitudes
            if calibrating:
                calibration_amplitudes.append(amplitude)
                if len(calibration_amplitudes) >= CALIBRATION_PACKETS:
                    baseline = np.mean(calibration_amplitudes)
                    calibrating = False
                    print(f"Baseline amplitude: {baseline:.1f}")
                    print("Ready! Block the signal path to input Morse code.")
                    print("-" * 50)
                continue

            # Update baseline with EMA (only when not blocked)
            if not blocked:
                baseline = (1 - EMA_ALPHA) * baseline + EMA_ALPHA * amplitude

            threshold = baseline * AMPLITUDE_DROP_FACTOR
            now = time.time()

            if amplitude < threshold and not blocked:
                # Signal just dropped - hand blocking
                blocked = True
                block_start = now

            elif amplitude >= threshold and blocked:
                # Signal restored - hand removed
                blocked = False
                duration = now - block_start
                last_unblock_time = now

                if duration < MIN_BLOCKAGE_DURATION:
                    # Too short, likely noise
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
                    # Very long block = letter boundary
                    if symbol_buffer:
                        letter = decode_morse(symbol_buffer)
                        print(letter, end="", flush=True)
                        symbol_buffer = ""
                    sys.stderr.write("|")
                    sys.stderr.flush()

            # Check for letter gap (unblocked for long enough)
            if not blocked and symbol_buffer:
                gap = now - last_unblock_time
                if gap > LETTER_GAP:
                    letter = decode_morse(symbol_buffer)
                    print(letter, end="", flush=True)
                    symbol_buffer = ""

    except KeyboardInterrupt:
        # Flush remaining symbols
        if symbol_buffer:
            letter = decode_morse(symbol_buffer)
            print(letter, end="", flush=True)
        print()
        print(f"\nTotal CSI packets processed: {packet_count}")
        print("Goodbye!")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
