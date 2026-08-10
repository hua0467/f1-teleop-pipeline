"""Pico UDP Sniffer — listen for any hand tracking data, log raw format.

Usage:
  python scripts/pico_sniffer.py --port 9000 --duration 10
  python scripts/pico_sniffer.py --ports 9000,9120,12345 --duration 30

This script auto-detects the format of incoming UDP data and logs samples.
Use it to figure out what format a new VR device sends before adapting the pipeline.
"""

import socket
import argparse
import time
import sys
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9000, help="Single port to listen on")
parser.add_argument("--ports", type=str, default=None, help="Comma-separated ports, e.g. '9000,9120'")
parser.add_argument("--duration", type=float, default=30.0, help="How long to listen (seconds)")
parser.add_argument("--output", type=str, default="./recordings/pico_sniff.log")
args = parser.parse_args()

# Determine ports to listen on
if args.ports:
    ports = [int(p.strip()) for p in args.ports.split(",")]
else:
    ports = [args.port]

print(f"[SNIFF] Listening on UDP ports {ports} for {args.duration}s...")
print(f"[SNIFF] Press Ctrl+C to stop early.\n")

# Open a socket for each port
sockets = {}
for port in ports:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.5)
        sockets[port] = sock
        print(f"  [OK] Port {port} bound")
    except Exception as e:
        print(f"  [FAIL] Port {port}: {e}")

if not sockets:
    print("[FAIL] No ports could be bound. Exiting.")
    sys.exit(1)

# Stats
packet_counts = defaultdict(int)
sample_packets = []  # first 20 packets for analysis
max_samples = 20
total_bytes = 0

t0 = time.time()
last_status = t0

try:
    while time.time() - t0 < args.duration:
        for port, sock in sockets.items():
            try:
                data, addr = sock.recvfrom(65535)
                packet_counts[port] += 1
                total_bytes += len(data)

                # Save first N packets as samples
                if len(sample_packets) < max_samples:
                    sample_packets.append({
                        "port": port,
                        "addr": addr,
                        "size": len(data),
                        "raw": data,
                        "time": time.time() - t0,
                    })

                # Status update every 2 seconds
                if time.time() - last_status > 2.0:
                    total_pkts = sum(packet_counts.values())
                    print(f"  [{time.time()-t0:.0f}s] {total_pkts} packets, {total_bytes/1024:.1f} KB", end="\r")
                    last_status = time.time()

            except socket.timeout:
                pass

except KeyboardInterrupt:
    print("\n[SNIFF] Stopped by user.")

finally:
    for sock in sockets.values():
        sock.close()

elapsed = time.time() - t0
total_pkts = sum(packet_counts.values())

print(f"\n\n{'='*60}")
print(f"[SNIFF] Results after {elapsed:.1f}s:")
print(f"  Total packets: {total_pkts}")
print(f"  Total data: {total_bytes/1024:.1f} KB")
for port, count in packet_counts.items():
    print(f"  Port {port}: {count} packets")
print()

if total_pkts == 0:
    print("[FAIL] ZERO packets received.")
    print("  Check: Pico WiFi connected? App started? IP correct?")
    print("  Try: ping <pico-ip> from this PC")
    sys.exit(1)

# Analyze samples
print(f"[ANALYZE] Examining {len(sample_packets)} sample packets...\n")

# Detect format by looking at first bytes
format_guess = "unknown"
for sp in sample_packets:
    raw = sp["raw"]

    # Try to decode as text
    try:
        text = raw.decode("utf-8", errors="replace")
        is_text = all(32 <= ord(c) < 127 or c in "\n\r\t" for c in text[:min(200, len(text))] if c not in "\n\r\t")
    except:
        text = None
        is_text = False

    # Try to detect protobuf/binary
    if raw[:4] == b"\x00\x00\x00\x00" or raw[0] < 0x08:
        format_guess = "binary/protobuf"
    elif is_text and "|" in (text or ""):
        format_guess = "text-keyvalue (like Quest3)"
    elif is_text and "{" in (text or ""):
        format_guess = "JSON"
    elif is_text:
        format_guess = "text-csv/plain"
    else:
        format_guess = "binary-unknown"

    break

print(f"  Detected format: {format_guess}")
print()

# Print sample details
for i, sp in enumerate(sample_packets[:10]):
    print(f"--- Sample {i+1} (port={sp['port']}, t={sp['time']:.2f}s, size={sp['size']}B, from={sp['addr']}) ---")
    raw = sp["raw"]
    try:
        text = raw.decode("utf-8", errors="replace")
        # Truncate for display
        if len(text) > 600:
            print(f"  TEXT[{len(text)} chars]: {text[:300]}")
            print(f"  ...{text[-300:]}")
        else:
            print(f"  TEXT: {text}")
    except:
        # Show hex dump for binary
        hex_str = raw[:128].hex()
        print(f"  HEX[{len(raw)}B]: {hex_str}")
        if len(raw) > 128:
            print(f"  ...({len(raw)-128} more bytes)")
    print()

# Save full log
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"# Pico UDP Sniff Log\n")
    f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# Duration: {elapsed:.1f}s, Packets: {total_pkts}\n")
    f.write(f"# Detected format: {format_guess}\n\n")
    for i, sp in enumerate(sample_packets[:20]):
        f.write(f"=== Sample {i+1} (port={sp['port']}, t={sp['time']:.2f}s, size={sp['size']}B) ===\n")
        try:
            f.write(sp["raw"].decode("utf-8", errors="replace") + "\n")
        except:
            f.write(sp["raw"].hex() + "\n")
        f.write("\n")

print(f"\n[LOG] Full samples saved to: {output_path}")
print(f"[NEXT] Based on the detected format, we'll write the adapter.")
print(f"[NEXT] If format={format_guess} matches Quest3, no changes needed!")
