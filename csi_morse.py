#!/usr/bin/env python3
"""WiFi CSI Hand-Gesture Morse Code Detector.

Uses Nexmon CSI on a Raspberry Pi 4B to detect hand blockages
in the WiFi signal path and decode them as Morse code.
Requires sudo.

CSI parsing adapted from github.com/jakka351/WIFI-CSI-Motion-Detection
"""

import subprocess
import struct
import time
import sys
import os
import numpy as np

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

DOT_MAX = 0.3
DASH_MAX = 0.8
LETTER_GAP = 0.8

AMPLITUDE_DROP_FACTOR = 0.7
EMA_ALPHA = 0.01
MIN_BLOCKAGE_DURATION = 0.05

INTERFACE = "wlan0"
NEXMON_HEADER_SIZE = 18
MAKECSIPARAMS = "/home/jonas/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams"

# Subcarrier mask for 256 subcarriers (80MHz)
fx256 = np.ones(256, dtype=bool)
fx256[:6] = False
fx256[64 * 4 - 5:] = False
fx256[32] = False
fx256[96] = False
fx256[160] = False
fx256[224] = False
for i in range(1, 4):
    fx256[64 * i - 5:64 * i + 6] = False

PCAP_GLOBAL_HDR = 24
PCAP_PKT_HDR = 16
ETH_HDR = 14
IP_HDR = 20
UDP_HDR = 8
FRAME_OVERHEAD = ETH_HDR + IP_HDR + UDP_HDR


def ensure_csi_active():
    """Ensure Nexmon CSI is configured and monitor mode is on."""
    result = subprocess.run(
        ["nexutil", f"-I{INTERFACE}", "-m"],
        capture_output=True, text=True, timeout=5,
    )
    if "monitor: 1" in result.stdout:
        print(f"CSI already active on {INTERFACE}", flush=True)
        return True

    print("Configuring CSI...", flush=True)

    # Unmanage from NetworkManager
    subprocess.run(["nmcli", "dev", "disconnect", INTERFACE],
                    check=False, capture_output=True, timeout=5)
    subprocess.run(["nmcli", "dev", "set", INTERFACE, "managed", "no"],
                    check=False, capture_output=True, timeout=5)

    subprocess.run(["ifconfig", INTERFACE, "up"],
                    check=False, capture_output=True, timeout=5)
    subprocess.run(["iw", "dev", INTERFACE, "set", "power_save", "off"],
                    check=False, capture_output=True, timeout=5)
    subprocess.run(["nexutil", "-s86", "-i", "-v0"],
                    check=False, capture_output=True, timeout=5)

    params = subprocess.run(
        [MAKECSIPARAMS, "-c", "149/80", "-C", "1", "-N", "1", "-b", "0x80"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()

    subprocess.run(
        ["nexutil", f"-I{INTERFACE}", "-s500", "-b", "-l34", f"-v{params}"],
        check=False, capture_output=True, timeout=15,
    )
    subprocess.run(
        ["nexutil", f"-I{INTERFACE}", "-m1"],
        check=False, capture_output=True, timeout=5,
    )

    result = subprocess.run(
        ["nexutil", f"-I{INTERFACE}", "-m"],
        capture_output=True, text=True, timeout=5,
    )
    if "monitor: 1" in result.stdout:
        print(f"CSI active on {INTERFACE} (channel 149/80MHz)", flush=True)
        return True
    print("WARNING: Could not enable monitor mode", flush=True)
    return False


def read_exactly(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_csi_payload(payload: bytes) -> np.ndarray | None:
    if len(payload) < NEXMON_HEADER_SIZE + 4:
        return None

    magic = struct.unpack(">H", payload[:2])[0]
    if magic != 0x1111:
        return None

    count = (len(payload) - NEXMON_HEADER_SIZE) // 2
    if count < 2:
        return None

    data = np.frombuffer(
        payload, dtype="<h", count=count, offset=NEXMON_HEADER_SIZE
    ).astype(np.float32).view(np.complex64)

    n_complex = len(data)

    if n_complex == 256:
        pl = data[fx256]
        x = pl.reshape((4, -1))
        mn = np.mean(np.abs(x), axis=-1)
        msk = np.zeros(mn.shape, dtype=bool)
        msk[np.argmax(mn)] = True
        pl = x[msk].ravel()
    else:
        skip = max(2, n_complex // 20)
        pl = data[skip:-skip] if n_complex > skip * 2 else data

    return pl


def compute_amplitude(csi: np.ndarray) -> float:
    return float(np.mean(np.abs(csi)))


def decode_morse(symbols: str) -> str:
    return MORSE_TO_CHAR.get(symbols, "?")


def main():
    print("=" * 50)
    print("  WiFi CSI Morse Code Detector")
    print("=" * 50, flush=True)

    ensure_csi_active()

    print(f"Dot: <{DOT_MAX}s  Dash: {DOT_MAX}-{DASH_MAX}s  Letter gap: >{LETTER_GAP}s")
    print(flush=True)

    proc = subprocess.Popen(
        ["tcpdump", "-i", INTERFACE, "-U", "-w", "-", "dst", "port", "5500"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    hdr = read_exactly(proc.stdout, PCAP_GLOBAL_HDR)
    if hdr is None:
        print("ERROR: Failed to start tcpdump.", flush=True)
        sys.exit(1)

    baseline = None
    blocked = False
    block_start = 0.0
    last_unblock_time = time.time()
    symbol_buffer = ""
    packet_count = 0
    calibration_amplitudes = []
    CALIBRATION_PACKETS = 50

    print("Calibrating baseline... keep hands clear of the signal path.", flush=True)

    try:
        while True:
            pkt_hdr = read_exactly(proc.stdout, PCAP_PKT_HDR)
            if pkt_hdr is None:
                print("\nCapture ended.", flush=True)
                break

            ts_sec, ts_usec, incl_len, orig_len = struct.unpack("<IIII", pkt_hdr)

            pkt_data = read_exactly(proc.stdout, incl_len)
            if pkt_data is None:
                break

            if incl_len < FRAME_OVERHEAD:
                continue
            payload = pkt_data[FRAME_OVERHEAD:]

            csi = parse_csi_payload(payload)
            if csi is None:
                continue

            amplitude = compute_amplitude(csi)
            packet_count += 1

            if len(calibration_amplitudes) < CALIBRATION_PACKETS:
                calibration_amplitudes.append(amplitude)
                if len(calibration_amplitudes) == CALIBRATION_PACKETS:
                    baseline = np.mean(calibration_amplitudes)
                    print(f"Baseline amplitude: {baseline:.1f}", flush=True)
                    print("Ready! Block the signal path to input Morse code.", flush=True)
                    print("-" * 50, flush=True)
                continue

            if not blocked:
                baseline = (1 - EMA_ALPHA) * baseline + EMA_ALPHA * amplitude

            threshold = baseline * AMPLITUDE_DROP_FACTOR
            now = time.time()

            if amplitude < threshold and not blocked:
                blocked = True
                block_start = now

            elif amplitude >= threshold and blocked:
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

            if not blocked and symbol_buffer:
                gap = now - last_unblock_time
                if gap > LETTER_GAP:
                    letter = decode_morse(symbol_buffer)
                    print(letter, end="", flush=True)
                    symbol_buffer = ""

    except KeyboardInterrupt:
        if symbol_buffer:
            letter = decode_morse(symbol_buffer)
            print(letter, end="", flush=True)
        print()
        print(f"\nTotal CSI packets processed: {packet_count}")
        print("Goodbye!")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
