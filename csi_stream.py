#!/usr/bin/env python3
"""CSI stream adapter for Nexmon CSI on Makefile.rpi setups.

Captures CSI via tcpdump (firmware-injected packets) and outputs
CSV lines: mac,rssi,motion — compatible with jakka351's matrix.py.

Usage:
    sudo python3 -u csi_stream.py | python3 ../WIFI-CSI-Motion-Detection/matrix.py
"""

import subprocess
import struct
import sys
import numpy as np

INTERFACE = "wlan0"
NEXMON_HEADER_SIZE = 18
PCAP_GLOBAL_HDR = 24
PCAP_PKT_HDR = 16
ETH_HDR = 14
IP_HDR = 20
UDP_HDR = 8
FRAME_OVERHEAD = ETH_HDR + IP_HDR + UDP_HDR

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


def main():
    proc = subprocess.Popen(
        ["tcpdump", "-i", INTERFACE, "-U", "-w", "-", "dst", "port", "5500"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    hdr = read_exactly(proc.stdout, PCAP_GLOBAL_HDR)
    if hdr is None:
        print("ERROR: tcpdump failed", file=sys.stderr)
        sys.exit(1)

    prev = None
    cnt = 0

    try:
        while True:
            pkt_hdr = read_exactly(proc.stdout, PCAP_PKT_HDR)
            if pkt_hdr is None:
                break

            _, _, incl_len, _ = struct.unpack("<IIII", pkt_hdr)
            pkt_data = read_exactly(proc.stdout, incl_len)
            if pkt_data is None:
                break

            if incl_len < FRAME_OVERHEAD + NEXMON_HEADER_SIZE + 4:
                continue

            payload = pkt_data[FRAME_OVERHEAD:]
            magic = struct.unpack(">H", payload[:2])[0]
            if magic != 0x1111:
                continue

            # Parse Nexmon header
            _, _, rssi, _, mac_bytes, _, _, _, _ = struct.unpack(
                "<BBbB6sHHHH", payload[:NEXMON_HEADER_SIZE]
            )
            mac_str = mac_bytes.hex(":")

            count = (len(payload) - NEXMON_HEADER_SIZE) // 2
            if count < 2:
                continue

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

            v = np.abs(pl)
            maxv = np.max(v)
            if maxv != 0:
                v /= maxv

            if prev is not None:
                corr = np.corrcoef(v, prev)[0][1]
                motion = -10 * corr ** 2 + 10
                print(f"{mac_str},{rssi},{motion}", flush=True)

            prev = v
            cnt += 1

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
