#!/usr/bin/env python3
"""Real-time CSI viewer — runs on your Mac, streams from Pi via SSH.

Usage:
    python3 csi_realtime_viewer.py [pi-address]

    Default Pi address: 192.168.1.237
    Requires: pip install matplotlib numpy scipy

The script SSHs into the Pi, runs tcpdump to capture Nexmon CSI
packets, and plots them live with matplotlib.

Panels:
  1. Subcarrier amplitudes (bar chart)
  2. Amplitude heatmap (time x subcarrier waterfall)
  3. STFT spectrogram (frequency of motion over time)
  4. Motion energy + presence indicator
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

HISTORY_LEN = 300
STFT_WINDOW = 64
STFT_HOP = 4
UPDATE_INTERVAL_MS = 80

MOTION_THRESHOLD = 0.3
PRESENCE_WINDOW = 15

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


def compute_stft(signal, window_size, hop):
    """Compute STFT magnitude on a 1D signal. Returns 2D array (freq x time)."""
    n = len(signal)
    n_frames = max(0, (n - window_size) // hop + 1)
    if n_frames < 1:
        return np.zeros((window_size // 2, 1))

    win = np.hanning(window_size)
    result = np.zeros((window_size // 2, n_frames))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + window_size] * win
        spectrum = np.abs(np.fft.rfft(frame))
        result[:, i] = spectrum[1:]  # drop DC

    return result


def main():
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors

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
    variance_history = np.zeros(HISTORY_LEN)
    prev_csi = [None]
    rssi_val = [0]
    mac_val = [""]
    pkt_count = [0]
    start_time = [time.time()]

    # Set up plots — 4 panels
    fig = plt.figure(figsize=(16, 11), facecolor="#0a0a1a")
    fig.canvas.manager.set_window_title("WiFi CSI Real-Time Viewer")
    gs = GridSpec(4, 1, height_ratios=[1.5, 1.5, 1.5, 1], hspace=0.4, figure=fig)

    ax_bar = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_spec = fig.add_subplot(gs[2])
    ax_motion = fig.add_subplot(gs[3])

    for ax in [ax_bar, ax_heat, ax_spec, ax_motion]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#666", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#222")

    status_text = fig.text(0.5, 0.99, "", ha="center", va="top",
                            color="#00ff88", fontsize=10, family="monospace")
    presence_text = fig.text(0.95, 0.99, "", ha="right", va="top",
                              fontsize=14, family="monospace", weight="bold")

    def read_packets():
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

            amp_history[:-1] = amp_history[1:]
            amp_history[-1] = amplitudes

            mean_history[:-1] = mean_history[1:]
            mean_history[-1] = np.mean(amplitudes)

            # Sliding window variance (motion sensitivity)
            win = min(10, pkt_count[0])
            if win > 1:
                recent = mean_history[-win:]
                variance_history[-1] = np.var(recent)
            variance_history[:-1] = variance_history[1:]

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

        # --- Panel 1: Subcarrier amplitudes ---
        ax_bar.cla()
        ax_bar.set_facecolor("#0d1117")
        norm_amp = amp / (np.max(amp) + 1e-9)
        colors = plt.cm.plasma(norm_amp)
        ax_bar.bar(range(ns), amp, color=colors, width=1.0, edgecolor="none")
        ax_bar.set_xlim(0, ns)
        ax_bar.set_title("Subcarrier Amplitudes", color="#c0c0c0", fontsize=11, loc="left")
        ax_bar.tick_params(colors="#666", labelsize=8)

        # --- Panel 2: Amplitude heatmap waterfall ---
        ax_heat.cla()
        ax_heat.set_facecolor("#0d1117")
        ax_heat.imshow(amp_history, aspect="auto", cmap="inferno",
                        interpolation="bilinear", origin="lower")
        ax_heat.set_title("Amplitude Waterfall (time × subcarrier)", color="#c0c0c0",
                          fontsize=11, loc="left")
        ax_heat.set_ylabel("Time →", color="#666", fontsize=9)
        ax_heat.tick_params(colors="#666", labelsize=8)

        # --- Panel 3: STFT spectrogram of mean amplitude ---
        ax_spec.cla()
        ax_spec.set_facecolor("#0d1117")

        spec = compute_stft(mean_history, STFT_WINDOW, STFT_HOP)
        if spec.size > 0:
            # Estimate frequency axis (assuming ~10 packets/sec)
            pps_est = pkt_count[0] / max(0.1, time.time() - start_time[0])
            freq_max = pps_est / 2.0
            n_freq = spec.shape[0]
            # Only show low frequencies (0-5 Hz) where motion lives
            max_freq_idx = min(n_freq, max(1, int(5.0 / freq_max * n_freq))) if freq_max > 0 else n_freq
            spec_show = spec[:max_freq_idx, :]

            # Log scale for better contrast
            spec_db = 10 * np.log10(spec_show + 1e-10)
            vmin = np.percentile(spec_db, 10)
            vmax = np.percentile(spec_db, 99)

            ax_spec.imshow(spec_db, aspect="auto", cmap="magma",
                            interpolation="bilinear", origin="lower",
                            vmin=vmin, vmax=vmax,
                            extent=[0, spec_db.shape[1], 0, min(5.0, freq_max)])
            ax_spec.set_ylabel("Freq (Hz)", color="#666", fontsize=9)

        ax_spec.set_title("Motion Spectrogram (STFT)", color="#c0c0c0",
                          fontsize=11, loc="left")
        ax_spec.tick_params(colors="#666", labelsize=8)

        # --- Panel 4: Motion energy + presence ---
        ax_motion.cla()
        ax_motion.set_facecolor("#0d1117")
        x = np.arange(HISTORY_LEN)

        # Color-coded motion fill
        ax_motion.fill_between(x, motion_history, alpha=0.4, color="#00ff88")
        ax_motion.plot(x, motion_history, color="#00ff88", linewidth=1.5)

        # Threshold line
        ax_motion.axhline(y=MOTION_THRESHOLD, color="#ff4444", linewidth=0.8,
                          linestyle="--", alpha=0.6)

        m_max = max(np.max(motion_history), 1.0)
        ax_motion.set_xlim(0, HISTORY_LEN)
        ax_motion.set_ylim(0, m_max * 1.1)
        ax_motion.set_title("Motion Energy", color="#c0c0c0", fontsize=11, loc="left")
        ax_motion.set_xlabel("Packets ago", color="#666", fontsize=9)
        ax_motion.tick_params(colors="#666", labelsize=8)

        # Presence detection
        recent_motion = motion_history[-PRESENCE_WINDOW:]
        presence = np.mean(recent_motion) > MOTION_THRESHOLD
        if presence:
            presence_text.set_text("● MOTION")
            presence_text.set_color("#ff4444")
        else:
            presence_text.set_text("● STILL")
            presence_text.set_color("#00cc66")

        # Status bar
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
