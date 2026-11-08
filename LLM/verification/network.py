"""Kernel network evidence for trusted offline workers, including Synology.

Names alone are not proof. A dormant IPv6-in-IPv4 fallback device is accepted
only with matching kernel indices, no UP/RUNNING flags, no IP and no route.
ARPHRD_SIT=776 is defined by Linux include/uapi/linux/if_arp.h.
"""
from pathlib import Path
import socket

import psutil

VERSION = 'kernel-offline-network-v1'


def collect():
    sysfs = Path('/sys/class/net')
    indices = dict((name, index) for index, name in socket.if_nameindex())
    proc = Path('/proc/net/dev').read_text().splitlines()
    devices = {line.split(':', 1)[0].strip() for line in proc[2:] if ':' in line}
    links = {}
    metadata = []
    for entry in sysfs.iterdir():
        index = entry / 'ifindex'
        if not index.is_file():
            if entry.name == 'syno_ovs_bonds' and entry.name not in indices and entry.name not in devices:
                metadata.append(entry.name)
                continue
            raise ValueError('Unrecognized sysfs network entry')
        links[entry.name] = {'index':int(index.read_text()),
            'type':int((entry / 'type').read_text()),
            'flags':int((entry / 'flags').read_text().strip(), 16)}
    addresses = psutil.net_if_addrs()
    ip_devices = {name for name, values in addresses.items()
                  if any(value.family in (socket.AF_INET, socket.AF_INET6) for value in values)}
    route4 = Path('/proc/net/route').read_text().splitlines()
    if not route4 or not route4[0].split() or route4[0].split()[0] != 'Iface':
        raise ValueError('Invalid IPv4 route table')
    routes = []
    for line in route4[1:]:
        if not line.strip():continue
        fields = line.split()
        if len(fields) < 11:raise ValueError('Invalid IPv4 route')
        routes.append(fields[0])
    route6 = Path('/proc/net/ipv6_route').read_text().splitlines()
    for line in route6:
        if not line.strip():continue
        fields = line.split()
        if len(fields) != 10:raise ValueError('Invalid IPv6 route')
        routes.append(fields[-1])
    return {'indices':indices, 'devices':sorted(devices), 'links':links,
            'ipDevices':sorted(ip_devices), 'routeDevices':routes, 'metadata':metadata}


def evaluate(snapshot):
    names = set(snapshot['indices'])
    reasons = []
    if not names or 'lo' not in names or names != set(snapshot['devices']) or names != set(snapshot['links']):
        reasons.append('NETWORK_INTERFACE_VIEWS_DIFFER')
    for name, link in snapshot['links'].items():
        if link['index'] != snapshot['indices'].get(name):
            reasons.append('NETWORK_INTERFACE_INDEX_MISMATCH')
        if name == 'lo':
            if link['type'] != 772:reasons.append('NETWORK_LOOPBACK_TYPE_INVALID')
        elif (name != 'sit0' or link['type'] != 776 or link['flags'] & (1 | 64)):
            reasons.append('NETWORK_EXTERNAL_LINK_PRESENT')
    if set(snapshot['ipDevices']) - {'lo'}:
        reasons.append('NETWORK_EXTERNAL_IP_PRESENT')
    if set(snapshot['routeDevices']) - {'lo'}:
        reasons.append('NETWORK_EXTERNAL_ROUTE_PRESENT')
    return {'version':VERSION, 'isolated':not reasons, 'interfaces':sorted(names),
            'ignoredMetadata':snapshot['metadata'], 'reasonCodes':sorted(set(reasons))}


def evidence():
    try:return evaluate(collect())
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, psutil.Error):
        return {'version':VERSION, 'isolated':False, 'reasonCodes':['NETWORK_KERNEL_EVIDENCE_UNAVAILABLE']}
