"""Read-only NIC/PCIe and physical GPU usage; no CUDA or persistent history."""
import csv
import math
from pathlib import Path
import re
import socket
import subprocess
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def number(value):
    match = re.match(r'^\s*(\d+(?:\.\d+)?)', str(value or ''))
    if not match:
        return None
    value = float(match[1])
    return value if math.isfinite(value) and value > 0 else None


def measurement(value, maximum=None):
    """Accept real zero readings while keeping N/A and invalid readings null."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value >= 0 and (maximum is None or value <= maximum) else None


def generation(speed):
    # Exact negotiated transfer rates; unknown values never imply a generation.
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6, 128.0: 7}.get(speed)


def command(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=3)
        return proc.stdout if proc.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired):
        return ''


def link(device):
    speed = number(read(device / 'current_link_speed'))
    maximum = number(read(device / 'max_link_speed'))
    return dict(link_speed_gt_s=speed, link_width=number(read(device / 'current_link_width')),
                pcie_generation=generation(speed), max_link_speed_gt_s=maximum,
                max_link_width=number(read(device / 'max_link_width')),
                max_pcie_generation=generation(maximum))


def model_name(description, vendor, device):
    if description:
        match = re.search(r'\[([^\]]+)\]', description)
        return match[1] if match else re.sub(r'^[^:]+:\s*', '', description)
    # Match PCI IDs, never hostnames, partitions, or Ethernet port rates.
    known = {'0x1017': 'ConnectX-5', '0x1019': 'ConnectX-5 Ex', '0x101d': 'ConnectX-6 Dx'}
    if vendor == '0x15b3' and device in known:
        return known[device]
    return ' / '.join(item for item in (vendor, device) if item) or None


def nics(sysfs=Path('/sys'), descriptions=None):
    descriptions = descriptions or {}
    rows, seen = [], set()
    for hca in sorted((sysfs / 'class/infiniband').glob('*')):
        device = (hca / 'device').resolve()
        if not (hca / 'device').exists() or device.name in seen:
            continue
        seen.add(device.name)
        vendor, ident = read(device / 'vendor'), read(device / 'device')
        ancestors = [dict(pci_address=p.name, **link(p)) for p in device.parents
                     if (p / 'current_link_speed').exists()]
        rows.append(dict(hca=hca.name, pci_address=device.name, vendor=vendor, device=ident,
                         model=model_name(descriptions.get(device.name), vendor, ident),
                         ancestors=ancestors, **link(device)))
    return rows


def gpus(text):
    rows = []
    for fields in csv.reader(text.splitlines()):
        if len(fields) not in (6, 9):
            continue
        fields = [s.strip() for s in fields]
        index, pci, current, maximum, width, max_width = fields[:6]
        if not index.isdigit() or any(row['index'] == int(index) for row in rows):
            continue
        rows.append(dict(index=int(index), pci_address=pci,
                         pcie_generation=number(current), max_pcie_generation=number(maximum),
                         pcie_width=number(width), max_pcie_width=number(max_width),
                         utilization_gpu_pct=measurement(fields[6], 100) if len(fields) == 9 else None,
                         memory_used_mib=measurement(fields[7]) if len(fields) == 9 else None,
                         memory_total_mib=measurement(fields[8]) if len(fields) == 9 else None))
    return rows


def probe():
    raw = command(['lspci', '-D'])
    descriptions = dict(line.split(' ', 1) for line in raw.splitlines() if ' ' in line)
    cards = nics(descriptions=descriptions)
    gpu_text = command(['nvidia-smi',
        '--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'])
    gpu_rows = gpus(gpu_text)
    return dict(status='current' if cards or gpu_rows else 'unknown',
                node=socket.gethostname().split('.')[0], sampled_at=time.time(),
                boot_id=read('/proc/sys/kernel/random/boot_id'),
                source='live_sysfs_lspci_nvidia_smi', nics=cards, gpus=gpu_rows)
