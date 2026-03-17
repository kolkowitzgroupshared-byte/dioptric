import ipaddress
import socket

CRLF = "\r\n"

def scpi_query(ip, cmd, port=5025, timeout=0.25):
    with socket.create_connection((str(ip), port), timeout=timeout) as s:
        s.sendall((cmd + CRLF).encode())
        s.settimeout(timeout)
        data = s.recv(4096)
    return data.decode(errors="ignore").strip()

def find_srs_sg390(subnet_cidr="192.168.0.0/24"):
    subnet = ipaddress.ip_network(subnet_cidr, strict=False)
    found = []
    for ip in subnet.hosts():
        try:
            idn = scpi_query(ip, "*IDN?")
            if "Stanford" in idn or "SRS" in idn or "SG39" in idn:
                mac = scpi_query(ip, "EMAC?")
                ip_addr = scpi_query(ip, "IPCF? 1")
                found.append((str(ip), ip_addr, mac, idn))
        except Exception:
            pass
    return found



def emac(ip, port=5025, timeout=0.3):
    with socket.create_connection((ip, port), timeout=timeout) as s:
        s.sendall(b"EMAC?\r\n")
        return s.recv(256).decode().strip()

for ip in ["192.168.1.10","192.168.1.11","192.168.1.12","192.168.1.13"]:
    print(ip, emac(ip))

import socket

IPS = [
    "192.168.0.178",
    "192.168.0.177",
    "192.168.0.174",
    "192.168.0.119",
    "192.168.0.121",
    "192.168.0.120",
]

def scpi_query(ip, cmd, port, timeout=0.4):
    with socket.create_connection((ip, port), timeout=timeout) as s:
        s.sendall((cmd + "\r\n").encode())
        s.settimeout(timeout)
        return s.recv(1024).decode(errors="ignore").strip()

def try_idn_emac(ip):
    # SG390-series supports: raw socket on 5025, telnet on 5024 (if enabled). :contentReference[oaicite:2]{index=2}
    for port in (5025, 5024):
        try:
            idn  = scpi_query(ip, "*IDN?", port)
            emac = scpi_query(ip, "EMAC?", port)
            return port, idn, emac
        except OSError:
            pass
    return None, None, None

for ip in IPS:
    port, idn, emac = try_idn_emac(ip)
    if port:
        print(f"{ip} (port {port})  IDN={idn}  EMAC={emac}")
    else:
        print(f"{ip}: no response on 5025/5024 (check NET menu enables)")

# if __name__ == "__main__":
#     for ip, ip_addr, mac, idn in find_srs_sg390("192.168.0.0/24"):
#         print(f"scanner_ip={ip}  reported_ip={ip_addr}  mac={mac}  idn={idn}")
