#!/usr/bin/env python3
"""CSI streaming server — runs on Pi, broadcasts parsed CSI over TCP.

Captures Nexmon CSI via tcpdump, parses packets, and streams JSON lines
to all connected clients. Auto-configures CSI on startup.

Usage:
    sudo python3 -u csi_server.py [--port 5555] [--host 0.0.0.0]

Clients connect via TCP and receive newline-delimited JSON:
    {"rssi": -45, "mac": "aa:bb:cc:dd:ee:ff", "n_sub": 234,
     "amplitudes": [...], "motion": 0.32, "seq": 1234, "t": 1234567890.123}
"""

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import time
import numpy as np

INTERFACE = "wlan0"
NEXMON_HEADER_SIZE = 18
MAKECSIPARAMS = "/home/jonas/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams"

PCAP_GLOBAL_HDR = 24
PCAP_PKT_HDR = 16
ETH_HDR = 14
IP_HDR = 20
UDP_HDR = 8
FRAME_OVERHEAD = ETH_HDR + IP_HDR + UDP_HDR

fx256 = np.ones(256, dtype=bool)
fx256[:6] = False
fx256[64 * 4 - 5:] = False
fx256[32] = False
fx256[96] = False
fx256[160] = False
fx256[224] = False
for i in range(1, 4):
    fx256[64 * i - 5:64 * i + 6] = False


def ensure_csi():
    r = subprocess.run(["nexutil", f"-I{INTERFACE}", "-m"],
                        capture_output=True, text=True, timeout=5)
    if "monitor: 1" in r.stdout:
        print(f"CSI already active on {INTERFACE}", flush=True)
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
    print("CSI activated.", flush=True)


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
    seq >>= 4

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

    return rssi, mac, seq, np.abs(pl)


class CSIServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.clients: set[asyncio.StreamWriter] = set()
        self.prev_csi = None
        self.pkt_count = 0
        self.start_time = time.time()

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info("peername")
        print(f"Client connected: {addr}", flush=True)
        self.clients.add(writer)
        try:
            await reader.read()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self.clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"Client disconnected: {addr}", flush=True)

    async def broadcast(self, data: bytes):
        dead = []
        for writer in self.clients:
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionError, asyncio.CancelledError):
                dead.append(writer)
        for w in dead:
            self.clients.discard(w)
            try:
                w.close()
            except Exception:
                pass

    async def capture_loop(self):
        proc = await asyncio.create_subprocess_exec(
            "tcpdump", "-i", INTERFACE, "-U", "-w", "-", "dst", "port", "5500",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        hdr = await proc.stdout.readexactly(PCAP_GLOBAL_HDR)
        if not hdr:
            print("ERROR: tcpdump failed to start", flush=True)
            return

        print("Capturing CSI packets...", flush=True)

        try:
            while True:
                pkt_hdr = await proc.stdout.readexactly(PCAP_PKT_HDR)
                _, _, incl_len, _ = struct.unpack("<IIII", pkt_hdr)
                pkt_data = await proc.stdout.readexactly(incl_len)

                if incl_len < FRAME_OVERHEAD + NEXMON_HEADER_SIZE:
                    continue

                payload = pkt_data[FRAME_OVERHEAD:]
                result = parse_csi(payload)
                if result is None:
                    continue

                rssi, mac, seq, amplitudes = result
                self.pkt_count += 1

                v = amplitudes / (np.max(amplitudes) + 1e-9)
                motion = 0.0
                if self.prev_csi is not None and len(v) == len(self.prev_csi):
                    corr = np.corrcoef(v, self.prev_csi)[0][1]
                    motion = float(-10 * corr ** 2 + 10)
                self.prev_csi = v

                msg = {
                    "rssi": int(rssi),
                    "mac": mac,
                    "seq": seq,
                    "n_sub": len(amplitudes),
                    "amplitudes": amplitudes.tolist(),
                    "motion": round(motion, 4),
                    "t": round(time.time(), 3),
                }
                line = json.dumps(msg, separators=(",", ":")) + "\n"
                await self.broadcast(line.encode())

                if self.pkt_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    pps = self.pkt_count / max(0.1, elapsed)
                    print(f"  {self.pkt_count} packets | {pps:.0f} pkt/s | "
                          f"{len(self.clients)} clients", flush=True)

        except (asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        finally:
            proc.terminate()
            await proc.wait()

    async def run(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
        print(f"CSI server listening on {addrs}", flush=True)
        print(f"Connect with: nc {self.host} {self.port}", flush=True)

        async with server:
            await asyncio.gather(
                server.serve_forever(),
                self.capture_loop(),
            )


def main():
    parser = argparse.ArgumentParser(description="CSI streaming server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--no-setup", action="store_true",
                        help="Skip CSI auto-configuration")
    args = parser.parse_args()

    if not args.no_setup:
        ensure_csi()

    srv = CSIServer(args.host, args.port)
    try:
        asyncio.run(srv.run())
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)


if __name__ == "__main__":
    main()
