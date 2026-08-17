# WiFi CSI Hand-Gesture Morse Code Detector

Detect hand gestures through WiFi signal disruption and decode them as Morse code — using a Raspberry Pi 4B with Nexmon CSI.

Block the WiFi signal path with your hand: short blocks become dots, longer blocks become dashes.

## Hardware

- Raspberry Pi 4B (BCM43455c0 WiFi chip)
- A second WiFi device (phone, laptop, or another Pi) as a traffic source
- Both devices on the same WiFi network

## Setup

### 1. Install Nexmon CSI

Follow the [Nexmon CSI guide](https://github.com/seemoo-lab/nexmon_csi) for the BCM43455c0 (Pi 4B).

```bash
# Install kernel headers
sudo apt install raspberrypi-kernel-headers

# Clone and build Nexmon CSI (see their README for full steps)
git clone https://github.com/seemoo-lab/nexmon_csi.git
cd nexmon_csi
# ... follow build instructions for bcm43455c0
```

After building, you'll have the patched firmware and the `nexutil`/`mcp` tools.

### 2. Start CSI Collection

On the Pi, activate the patched firmware and start streaming CSI to UDP:

```bash
# Load patched firmware (exact commands depend on your Nexmon build)
sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -s500 -b -l34 -v<your_chanspec_params>

# Start CSI streaming to UDP port 5500
sudo tcpdump -i wlan0 dst port 5500 -w - | nc -u localhost 5500 &
# OR use makecsiudp / nex_csi depending on your Nexmon version
```

### 3. Generate Traffic

From the transmitter device, flood the Pi with packets:

```bash
# Option A: ping flood (needs root)
sudo ping -f <pi-ip-address>

# Option B: iperf
iperf -c <pi-ip-address> -t 0 -u
```

### 4. Install Python Dependencies

```bash
pip install numpy
```

### 5. Run the Detector

```bash
python3 csi_morse.py
```

The script will:
1. Calibrate a baseline amplitude (keep hands clear for ~2 seconds)
2. Print decoded letters as you block the signal path

## How It Works

```
WiFi Transmitter  ~~~signal~~~  [hand blocks here]  ~~~signal~~~  Pi 4B
                                                                    |
                                                              Nexmon CSI
                                                                    |
                                                            UDP port 5500
                                                                    |
                                                           csi_morse.py
                                                                    |
                                                         amplitude drop?
                                                           /          \
                                                        yes            no
                                                         |              |
                                                   measure duration   update baseline
                                                    /     |      \
                                                <0.3s  0.3-0.8s  >0.8s
                                                  .       -      (letter gap)
```

## Morse Timing

| Duration | Meaning |
|----------|---------|
| < 0.3s | Dot (.) |
| 0.3–0.8s | Dash (-) |
| > 0.8s blockage | Letter separator |
| > 0.8s gap (no block) | Emit letter |
| > 2.0s gap | Word separator (space) |

## Tuning

Edit the constants at the top of `csi_morse.py`:

- `AMPLITUDE_DROP_FACTOR` — how much the signal must drop to count as blocked (default: 0.7 = 30% drop)
- `DOT_MAX` / `DASH_MAX` — timing boundaries for dot vs dash
- `EMA_ALPHA` — baseline adaptation speed (lower = slower adaptation)
- `MIN_BLOCKAGE_DURATION` — ignore blocks shorter than this (noise filter)

## Troubleshooting

**No CSI packets received:**
- Verify Nexmon CSI firmware is loaded: `nexutil -Iwlan0 -k`
- Check traffic is flowing: `tcpdump -i wlan0 -c 10`
- Ensure UDP port 5500 is receiving: `sudo tcpdump -i lo udp port 5500`

**False triggers / too sensitive:**
- Increase `AMPLITUDE_DROP_FACTOR` (e.g., 0.5 for a 50% drop requirement)
- Increase `MIN_BLOCKAGE_DURATION`

**Dots and dashes swapped or wrong:**
- Adjust `DOT_MAX` and `DASH_MAX` to match your hand speed
- Try practicing with deliberate short/long blocks

## License

MIT
