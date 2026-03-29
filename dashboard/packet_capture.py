from scapy.all import sniff
from datetime import datetime
import csv
import os

OUTPUT_FILE = "data/packets.csv"

def packet_callback(packet):
    if packet.haslayer("IP"):
        src = packet["IP"].src
        dst = packet["IP"].dst
        proto = packet["IP"].proto
        size = len(packet)
        time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        write_packet([time, src, dst, proto, size])

def write_packet(row):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "src_ip", "dst_ip", "protocol", "packet_size"]
            )
        writer.writerow(row)

print("Starting packet capture... Press CTRL+C to stop.")
sniff(prn=packet_callback, store=False)
