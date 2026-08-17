#!/usr/bin/env python3
"""WiFi CSI Hand-Gesture Morse Code Detector.

Uses Nexmon CSI on a Raspberry Pi 4B to detect hand blockages
in the WiFi signal path and decode them as Morse code.
Requires sudo.

Cover the Pi's antenna area with your hand to block the signal.
Short block = dot, long block = dash, pause = letter gap.

Usage: sudo python3 -u csi_morse.py
"""

import subprocess
import struct
import time
import sys
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
    ".-.-.-": ".", "--..--": ",", "..--..": "?",
}

DOT_MAX = 0.3
DASH_MAX = 1.0
LETTER_GAP = 1.0
WORD_GAP = 2.0
MIN_BLOCKAGE_DURATION = 0.05

BLOCK_THRESHOLD = 0.65
UNBLOCK_THRESHOLD = 0.80
BASELINE_EMA = 0.01
SIGNAL_EMA = 0.4
TOP_N_SUBCARRIERS = 20
CALIBRATION_PACKETS = 50

INTERFACE = "wlan0"
NEXMON_HEADER_SIZE = 18
MAKECSIPARAMS = "/home/jonas/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams"

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
    result = subprocess.run(
        ["nexutil", f"-I{INTERFACE}", "-m"],
        capture_output=True, text=True, timeout=5,
    )
    if "monitor: 1" in result.stdout:
        print(f"CSI already active on {INTERFACE}", flush=True)
        return True

    print("Configuring CSI...", flush=True)
    subprocess.run(["nmcli", "dev", "disconnect", INTERFACE],
                    check=False, capture_output=True, timeout=5)
    subprocess.run(["nmcli", "dev", "set", INTERFACE, "managed", "no"],
                    check=False, capture_output=True, timeout=5)
    subprocess.run(["ifconfig", INTERFACE, "up"],
                    check=False, capture_output=True, timeout=5)
    subprocess.run(["iw", "dev", INTERFACE, "set", "power_save", "off"],
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


def parse_csi_payload(payload):
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
    else:
        skip = max(1, n_complex // 20)
        pl = data[skip:-skip] if n_complex > skip * 2 else data

    return np.abs(pl)


def decode_morse(symbols):
    return MORSE_TO_CHAR.get(symbols, "?")


def main():
    print("=" * 50)
    print("  WiFi CSI Morse Code Detector")
    print("=" * 50, flush=True)

    ensure_csi_active()

    print(f"Timing:  dot <{DOT_MAX}s | dash {DOT_MAX}-{DASH_MAX}s | letter gap >{LETTER_GAP}s | word gap >{WORD_GAP}s")
    print(f"Thresholds:  block <{BLOCK_THRESHOLD:.0%} | unblock >{UNBLOCK_THRESHOLD:.0%} of baseline")
    print(f"Using top {TOP_N_SUBCARRIERS} subcarriers by signal strength")
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

    # Calibration state
    cal_samples = []
    best_subs = None
    baseline = None
    smoothed = None

    # Detection state
    blocked = False
    block_start = 0.0
    last_unblock_time = time.time()
    symbol_buffer = ""
    decoded_text = ""
    packet_count = 0

    print(f"Calibrating baseline ({CALIBRATION_PACKETS} packets)... keep hands clear.", flush=True)

    try:
        while True:
            pkt_hdr = read_exactly(proc.stdout, PCAP_PKT_HDR)
            if pkt_hdr is None:
                print("\nCapture ended.", flush=True)
                break

            _, _, incl_len, _ = struct.unpack("<IIII", pkt_hdr)
            pkt_data = read_exactly(proc.stdout, incl_len)
            if pkt_data is None:
                break
            if incl_len < FRAME_OVERHEAD:
                continue

            payload = pkt_data[FRAME_OVERHEAD:]
            amplitudes = parse_csi_payload(payload)
            if amplitudes is None:
                continue

            packet_count += 1

            # --- Calibration phase ---
            if len(cal_samples) < CALIBRATION_PACKETS:
                cal_samples.append(amplitudes)
                if len(cal_samples) == CALIBRATION_PACKETS:
                    stacked = np.array(cal_samples)
                    mean_per_sub = np.mean(stacked, axis=0)
                    best_subs = np.argsort(mean_per_sub)[-TOP_N_SUBCARRIERS:]
                    baseline = float(np.mean(mean_per_sub[best_subs]))
                    smoothed = baseline
                    print(f"Baseline: {baseline:.1f} (selected {len(best_subs)} subcarriers)", flush=True)
                    print("Ready! Cover the Pi to input Morse code.", flush=True)
                    print("-" * 50, flush=True)
                continue

            # --- Use selected subcarriers ---
            amp = float(np.mean(amplitudes[best_subs]))
            smoothed = SIGNAL_EMA * amp + (1 - SIGNAL_EMA) * smoothed

            now = time.time()

            # --- Hysteresis block detection ---
            if not blocked:
                baseline = (1 - BASELINE_EMA) * baseline + BASELINE_EMA * smoothed

                if smoothed < baseline * BLOCK_THRESHOLD:
                    blocked = True
                    block_start = now

            else:
                if smoothed > baseline * UNBLOCK_THRESHOLD:
                    blocked = False
                    duration = now - block_start
                    last_unblock_time = now

                    if duration < MIN_BLOCKAGE_DURATION:
                        pass
                    elif duration < DOT_MAX:
                        symbol_buffer += "."
                        sys.stderr.write("·")
                        sys.stderr.flush()
                    elif duration < DASH_MAX:
                        symbol_buffer += "-"
                        sys.stderr.write("—")
                        sys.stderr.flush()
                    else:
                        if symbol_buffer:
                            letter = decode_morse(symbol_buffer)
                            decoded_text += letter
                            print(letter, end="", flush=True)
                            symbol_buffer = ""
                        sys.stderr.write("|")
                        sys.stderr.flush()

            # --- Letter/word gap detection ---
            if not blocked and symbol_buffer:
                gap = now - last_unblock_time
                if gap > LETTER_GAP:
                    letter = decode_morse(symbol_buffer)
                    decoded_text += letter
                    print(letter, end="", flush=True)
                    symbol_buffer = ""

            if not blocked and decoded_text and not symbol_buffer:
                gap = now - last_unblock_time
                if gap > WORD_GAP and not decoded_text.endswith(" "):
                    decoded_text += " "
                    print(" ", end="", flush=True)

            # --- Live status on stderr ---
            if packet_count % 20 == 0:
                ratio = smoothed / baseline if baseline else 0
                state = "BLOCKED" if blocked else "open"
                bar_len = 30
                bar_fill = int(min(ratio, 1.2) / 1.2 * bar_len)
                bar = "█" * bar_fill + "░" * (bar_len - bar_fill)
                thresh_pos = int(BLOCK_THRESHOLD / 1.2 * bar_len)
                sys.stderr.write(
                    f"\r  [{bar}] {ratio:.0%} {state:>7}  "
                    f"buf:{''.join(symbol_buffer[-6:]): <6}"
                )
                sys.stderr.flush()

    except KeyboardInterrupt:
        if symbol_buffer:
            letter = decode_morse(symbol_buffer)
            decoded_text += letter
            print(letter, end="", flush=True)
        print()
        print(f"\nDecoded: {decoded_text}")
        print(f"Packets: {packet_count}")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
