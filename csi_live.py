#!/usr/bin/env python3
"""Live CSI terminal visualizer.

Shows real-time subcarrier amplitudes, signal strength, and motion
detection from Nexmon CSI packets. Requires sudo.

Usage: sudo python3 -u csi_live.py
"""

import subprocess
import struct
import sys
import time
import shutil
import signal
import numpy as np

INTERFACE = "wlan0"
NEXMON_HEADER_SIZE = 18
PCAP_GLOBAL_HDR = 24
PCAP_PKT_HDR = 16
ETH_HDR = 14
IP_HDR = 20
UDP_HDR = 8
FRAME_OVERHEAD = ETH_HDR + IP_HDR + UDP_HDR

TARGET_FPS = 15
EMA_ALPHA = 0.3

fx256 = np.ones(256, dtype=bool)
fx256[:6] = False
fx256[64 * 4 - 5:] = False
fx256[32] = False
fx256[96] = False
fx256[160] = False
fx256[224] = False
for i in range(1, 4):
    fx256[64 * i - 5:64 * i + 6] = False

BLOCKS = " ▁▂▃▄▅▆▇█"
MOTION_BAR = "░▒▓█"


def read_exactly(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def get_term_width():
    return shutil.get_terminal_size(fallback=(80, 24))


def color(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def bg_color(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"


RST = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
HOME = "\033[H"
CLEAR = "\033[J"


def amp_color(val):
    """Map 0..1 to blue->cyan->green->yellow->red."""
    if val < 0.25:
        t = val / 0.25
        return color(0, int(80 * t), int(200 + 55 * t))
    elif val < 0.5:
        t = (val - 0.25) / 0.25
        return color(0, int(80 + 175 * t), int(255 - 55 * t))
    elif val < 0.75:
        t = (val - 0.5) / 0.25
        return color(int(255 * t), 255, int(200 * (1 - t)))
    else:
        t = (val - 0.75) / 0.25
        return color(255, int(255 * (1 - t)), 0)


def motion_color(motion):
    if motion < 0.5:
        return color(60, 60, 60)
    elif motion < 2.0:
        return color(0, 200, 100)
    elif motion < 5.0:
        return color(255, 200, 0)
    else:
        return color(255, 50, 50)


class CSIVisualizer:
    def __init__(self):
        self.prev_csi = None
        self.smooth_amp = None
        self.motion_history = []
        self.amp_history = []
        self.max_amp = 1.0
        self.packet_count = 0
        self.start_time = time.time()
        self.last_draw = 0
        self.cols, self.rows = get_term_width()
        signal.signal(signal.SIGWINCH, lambda *_: self._resize())

    def _resize(self):
        self.cols, self.rows = get_term_width()

    def process(self, payload):
        if len(payload) < NEXMON_HEADER_SIZE + 4:
            return
        magic = struct.unpack(">H", payload[:2])[0]
        if magic != 0x1111:
            return

        _, _, rssi, _, mac_bytes, seq, _, chanspec, _ = struct.unpack(
            "<BBbB6sHHHH", payload[:NEXMON_HEADER_SIZE]
        )
        mac = mac_bytes.hex(":")
        seq >>= 4

        count = (len(payload) - NEXMON_HEADER_SIZE) // 2
        if count < 2:
            return

        data = np.frombuffer(
            payload, dtype="<h", count=count, offset=NEXMON_HEADER_SIZE
        ).astype(np.float32).view(np.complex64)

        n_complex = len(data)
        if n_complex == 256:
            pl = data[fx256]
        else:
            skip = max(1, n_complex // 20)
            pl = data[skip:-skip] if n_complex > skip * 2 else data

        amplitudes = np.abs(pl)

        if self.smooth_amp is None or len(self.smooth_amp) != len(amplitudes):
            self.smooth_amp = amplitudes.copy()
        else:
            self.smooth_amp = EMA_ALPHA * amplitudes + (1 - EMA_ALPHA) * self.smooth_amp

        motion = 0.0
        v = amplitudes / (np.max(amplitudes) + 1e-9)
        if self.prev_csi is not None and len(v) == len(self.prev_csi):
            corr = np.corrcoef(v, self.prev_csi)[0][1]
            motion = -10 * corr ** 2 + 10
        self.prev_csi = v

        mean_amp = float(np.mean(amplitudes))
        self.max_amp = max(self.max_amp * 0.999, float(np.max(amplitudes)))

        self.motion_history.append(motion)
        self.amp_history.append(mean_amp)
        max_hist = self.cols - 4
        if len(self.motion_history) > max_hist:
            self.motion_history = self.motion_history[-max_hist:]
        if len(self.amp_history) > max_hist:
            self.amp_history = self.amp_history[-max_hist:]

        self.packet_count += 1

        now = time.time()
        if now - self.last_draw < 1.0 / TARGET_FPS:
            return
        self.last_draw = now

        self.draw(mac, rssi, motion, mean_amp, seq)

    def draw(self, mac, rssi, motion, mean_amp, seq):
        w = self.cols
        out = [HOME]

        # Header
        elapsed = time.time() - self.start_time
        pps = self.packet_count / max(0.1, elapsed)
        out.append(f"{BOLD}{color(100, 200, 255)} WiFi CSI Live ")
        out.append(f"{RST}{DIM} {mac}  RSSI:{rssi}dBm  seq:{seq}  {pps:.0f}pkt/s  {self.packet_count}total{RST}\n")

        # Subcarrier amplitude bar chart
        amp = self.smooth_amp
        n = len(amp)
        bar_w = w - 2
        if n > bar_w:
            indices = np.linspace(0, n - 1, bar_w, dtype=int)
            amp_display = amp[indices]
        else:
            amp_display = amp
            bar_w = n

        norm = amp_display / (self.max_amp + 1e-9)
        out.append(f"{DIM}Subcarriers ({n}):{RST}\n")
        # 4 rows of bar chart
        for row in range(4, 0, -1):
            threshold = (row - 1) / 4.0
            line = []
            for i in range(len(amp_display)):
                v = float(norm[i])
                if v >= threshold:
                    level = min((v - threshold) * 4, 1.0)
                    bi = min(int(level * (len(BLOCKS) - 1)), len(BLOCKS) - 1)
                    line.append(amp_color(v) + BLOCKS[bi])
                else:
                    line.append(" ")
            out.append(" " + "".join(line) + RST + "\n")

        # Motion gauge
        out.append(f"\n{DIM}Motion:{RST} ")
        m_bar_w = min(40, w - 12)
        m_level = min(motion / 10.0, 1.0)
        m_filled = int(m_level * m_bar_w)
        out.append(motion_color(motion))
        for i in range(m_bar_w):
            if i < m_filled:
                bi = min(int((i / m_bar_w) * len(MOTION_BAR)), len(MOTION_BAR) - 1)
                out.append(MOTION_BAR[bi])
            else:
                out.append(f"{color(40, 40, 40)}·")
        out.append(f" {RST}{motion_color(motion)}{motion:.2f}{RST}\n")

        # Motion history sparkline
        out.append(f"\n{DIM}Motion history:{RST}\n ")
        if self.motion_history:
            max_m = max(max(self.motion_history), 0.1)
            for m in self.motion_history:
                v = min(m / max_m, 1.0)
                bi = min(int(v * (len(BLOCKS) - 1)), len(BLOCKS) - 1)
                out.append(motion_color(m) + BLOCKS[bi])
        out.append(RST + "\n")

        # Amplitude history sparkline
        out.append(f"\n{DIM}Amplitude history:{RST}\n ")
        if self.amp_history:
            max_a = max(max(self.amp_history), 0.1)
            min_a = min(self.amp_history)
            for a in self.amp_history:
                v = (a - min_a) / (max_a - min_a + 1e-9)
                bi = min(int(v * (len(BLOCKS) - 1)), len(BLOCKS) - 1)
                out.append(amp_color(v) + BLOCKS[bi])
        out.append(RST + "\n")

        # Status line
        out.append(f"\n{DIM}Press Ctrl+C to exit{RST}{CLEAR}")

        sys.stdout.write("".join(out))
        sys.stdout.flush()


MAKECSIPARAMS = "/home/jonas/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams"


def ensure_csi():
    """Activate Nexmon CSI if not already running."""
    r = subprocess.run(["nexutil", f"-I{INTERFACE}", "-m"],
                        capture_output=True, text=True, timeout=5)
    if "monitor: 1" in r.stdout:
        return

    print("Activating CSI...", flush=True)
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
    subprocess.run(["nexutil", f"-I{INTERFACE}", "-m1"],
                    check=False, capture_output=True, timeout=5)


def main():
    ensure_csi()

    sys.stdout.write(HIDE_CURSOR + HOME + CLEAR)
    sys.stdout.flush()

    viz = CSIVisualizer()

    proc = subprocess.Popen(
        ["tcpdump", "-i", INTERFACE, "-U", "-w", "-", "dst", "port", "5500"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    hdr = read_exactly(proc.stdout, PCAP_GLOBAL_HDR)
    if hdr is None:
        sys.stdout.write(SHOW_CURSOR)
        print("ERROR: No CSI packets. Try: nexutil -Iwlan0 -m1")
        sys.exit(1)

    try:
        while True:
            pkt_hdr = read_exactly(proc.stdout, PCAP_PKT_HDR)
            if pkt_hdr is None:
                break
            _, _, incl_len, _ = struct.unpack("<IIII", pkt_hdr)
            pkt_data = read_exactly(proc.stdout, incl_len)
            if pkt_data is None:
                break
            if incl_len < FRAME_OVERHEAD + NEXMON_HEADER_SIZE:
                continue
            payload = pkt_data[FRAME_OVERHEAD:]
            viz.process(payload)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW_CURSOR + RST + "\n")
        sys.stdout.flush()
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
