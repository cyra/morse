#!/usr/bin/env python3
"""Real-time CSI viewer — runs on your Mac, streams from Pi via SSH.

Usage:
    python3 csi_realtime_viewer.py [pi-address]

    Default Pi address: 192.168.1.237
    Requires: pip install matplotlib numpy

The script SSHs into the Pi, runs tcpdump to capture Nexmon CSI
packets, and plots them live with matplotlib.
"""

import subprocess
import struct
import sys
import time
import numpy as np

PI_HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.237"
PI_USER = "jonas"
INTERFACE = "wlan0"
NEXMON_HEADER_SIZE = 18

PCAP_GLOBAL_HDR = 24
PCAP_PKT_HDR = 16
ETH_HDR = 14
IP_HDR = 20
UDP_HDR = 8
FRAME_OVERHEAD = ETH_HDR + IP_HDR + UDP_HDR

HISTORY_LEN = 200
UPDATE_INTERVAL_MS = 80

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


def read_exactly(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_csi(payload):
    """Parse Nexmon CSI payload, return (rssi, mac, amplitudes, motion_csi)."""
    if len(payload) < NEXMON_HEADER_SIZE + 4:
        return None
    magic = struct.unpack(">H", payload[:2])[0]
    if magic != 0x1111:
        return None

    _, _, rssi, _, mac_bytes, seq, _, chanspec, _ = struct.unpack(
        "<BBbB6sHHHH", payload[:NEXMON_HEADER_SIZE]
    )
    mac = mac_bytes.hex(":")

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

    return rssi, mac, np.abs(pl)


def main():
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    print(f"Connecting to {PI_USER}@{PI_HOST}...")
    proc = subprocess.Popen(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{PI_USER}@{PI_HOST}",
            f"sudo tcpdump -i {INTERFACE} -U -w - dst port 5500",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    hdr = read_exactly(proc.stdout, PCAP_GLOBAL_HDR)
    if hdr is None:
        print("ERROR: Could not start capture. Is CSI active on the Pi?")
        sys.exit(1)
    print("Connected! Receiving CSI packets...")

    # State
    n_sub = [0]
    current_amp = [None]
    amp_history = np.zeros((HISTORY_LEN, 1))
    mean_history = np.zeros(HISTORY_LEN)
    motion_history = np.zeros(HISTORY_LEN)
    prev_csi = [None]
    rssi_val = [0]
    mac_val = [""]
    pkt_count = [0]
    start_time = [time.time()]

    # Set up plots
    fig = plt.figure(figsize=(14, 9), facecolor="#1a1a2e")
    fig.canvas.manager.set_window_title("WiFi CSI Real-Time Viewer")
    gs = GridSpec(3, 1, height_ratios=[2, 1.5, 1], hspace=0.35, figure=fig)

    ax_bar = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_motion = fig.add_subplot(gs[2])

    for ax in [ax_bar, ax_heat, ax_motion]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Bar chart
    bars_container = [None]
    ax_bar.set_title("Subcarrier Amplitudes", color="#e0e0e0", fontsize=12)
    ax_bar.set_xlabel("Subcarrier Index", color="#888")
    ax_bar.set_ylabel("Amplitude", color="#888")

    # Heatmap
    im = [None]
    ax_heat.set_title("Amplitude Heatmap (time × subcarrier)", color="#e0e0e0", fontsize=12)
    ax_heat.set_xlabel("Subcarrier Index", color="#888")
    ax_heat.set_ylabel("Time →", color="#888")

    # Motion plot
    motion_line, = ax_motion.plot([], [], color="#00ff88", linewidth=1.5)
    mean_line, = ax_motion.plot([], [], color="#4488ff", linewidth=1.2, alpha=0.7)
    ax_motion.set_title("Motion & Mean Amplitude", color="#e0e0e0", fontsize=12)
    ax_motion.set_xlabel("Packets ago", color="#888")
    ax_motion.legend(["Motion", "Mean Amp"], loc="upper left",
                      facecolor="#16213e", edgecolor="#333", labelcolor="#ccc")

    status_text = fig.text(0.5, 0.98, "", ha="center", va="top",
                            color="#00ff88", fontsize=10, family="monospace")

    def read_packets():
        """Non-blocking read of available packets."""
        count = 0
        while count < 20:
            pkt_hdr = read_exactly(proc.stdout, PCAP_PKT_HDR)
            if pkt_hdr is None:
                return False
            _, _, incl_len, _ = struct.unpack("<IIII", pkt_hdr)
            pkt_data = read_exactly(proc.stdout, incl_len)
            if pkt_data is None:
                return False
            if incl_len < FRAME_OVERHEAD + NEXMON_HEADER_SIZE:
                continue
            payload = pkt_data[FRAME_OVERHEAD:]
            result = parse_csi(payload)
            if result is None:
                continue

            rssi, mac, amplitudes = result
            rssi_val[0] = rssi
            mac_val[0] = mac
            current_amp[0] = amplitudes
            pkt_count[0] += 1

            ns = len(amplitudes)
            if ns != n_sub[0]:
                n_sub[0] = ns
                nonlocal amp_history
                amp_history = np.zeros((HISTORY_LEN, ns))

            # Shift history up and add new row at bottom
            amp_history[:-1] = amp_history[1:]
            amp_history[-1] = amplitudes

            mean_history[:-1] = mean_history[1:]
            mean_history[-1] = np.mean(amplitudes)

            # Motion via correlation
            v = amplitudes / (np.max(amplitudes) + 1e-9)
            motion = 0.0
            if prev_csi[0] is not None and len(v) == len(prev_csi[0]):
                corr = np.corrcoef(v, prev_csi[0])[0][1]
                motion = -10 * corr ** 2 + 10
            prev_csi[0] = v

            motion_history[:-1] = motion_history[1:]
            motion_history[-1] = motion

            count += 1
        return True

    def update(frame_num):
        if not read_packets():
            return

        if current_amp[0] is None or n_sub[0] == 0:
            return

        ns = n_sub[0]
        amp = current_amp[0]

        # Bar chart
        ax_bar.cla()
        ax_bar.set_facecolor("#16213e")
        colors = plt.cm.viridis(amp / (np.max(amp) + 1e-9))
        ax_bar.bar(range(ns), amp, color=colors, width=1.0)
        ax_bar.set_xlim(0, ns)
        ax_bar.set_title("Subcarrier Amplitudes", color="#e0e0e0", fontsize=12)
        ax_bar.tick_params(colors="#aaa")

        # Heatmap
        ax_heat.cla()
        ax_heat.set_facecolor("#16213e")
        ax_heat.imshow(amp_history, aspect="auto", cmap="inferno",
                        interpolation="nearest", origin="lower")
        ax_heat.set_title("Amplitude Heatmap (time × subcarrier)",
                          color="#e0e0e0", fontsize=12)
        ax_heat.tick_params(colors="#aaa")

        # Motion + mean amplitude
        ax_motion.cla()
        ax_motion.set_facecolor("#16213e")
        x = np.arange(HISTORY_LEN)

        m_max = max(np.max(motion_history), 0.1)
        a_max = max(np.max(mean_history), 0.1)
        ax_motion.fill_between(x, motion_history / m_max, alpha=0.3, color="#00ff88")
        ax_motion.plot(x, motion_history / m_max, color="#00ff88", linewidth=1.5, label="Motion")
        ax_motion.plot(x, mean_history / a_max, color="#4488ff", linewidth=1.2,
                        alpha=0.7, label="Mean Amp")
        ax_motion.set_xlim(0, HISTORY_LEN)
        ax_motion.set_ylim(0, 1.1)
        ax_motion.set_title("Motion & Mean Amplitude (normalized)",
                            color="#e0e0e0", fontsize=12)
        ax_motion.legend(loc="upper left", facecolor="#16213e",
                          edgecolor="#333", labelcolor="#ccc")
        ax_motion.tick_params(colors="#aaa")

        # Status
        elapsed = time.time() - start_time[0]
        pps = pkt_count[0] / max(0.1, elapsed)
        status_text.set_text(
            f"MAC: {mac_val[0]}  RSSI: {rssi_val[0]}dBm  "
            f"{pps:.0f} pkt/s  {pkt_count[0]} total  "
            f"Motion: {motion_history[-1]:.2f}"
        )

    ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL_MS, cache_frame_data=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
