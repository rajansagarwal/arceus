#!/usr/bin/env python3
"""
Debug script to understand exactly what _pick_macos_iface is doing
"""

import subprocess
import re
import socket
import ipaddress
import platform

def debug_pick_macos_iface():
    """Debug version of _pick_macos_iface that shows all steps"""
    print("🔍 Debugging _pick_macos_iface selection process")
    print("=" * 60)
    
    if platform.system() != "Darwin":
        print("❌ Not macOS, returning default")
        return "en0", "127.0.0.1"
    
    try:
        print("\n1. Getting ifconfig output...")
        ifconfig_out = subprocess.check_output(["ifconfig"]).decode()
        
        print("\n2. Finding UP interfaces...")
        up_interfaces = re.findall(r"^(en\d+):.*?<UP,.*?>", ifconfig_out, re.M)
        print(f"   Found {len(up_interfaces)} UP interfaces:")
        for iface in up_interfaces:
            print(f"     {iface}")
        
        print("\n3. Matching interfaces with their IPs...")
        candidates = []
        
        for iface in up_interfaces:
            print(f"\n   Processing {iface}:")
            
            # For each UP interface, find its IPv4 address
            # Look for the interface block and extract the inet address
            iface_pattern = re.compile(rf"^{re.escape(iface)}:.*?(?=^[a-zA-Z]|\Z)", re.M | re.S)
            iface_block = iface_pattern.search(ifconfig_out)
            
            if not iface_block:
                print(f"     ❌ Could not find interface block")
                continue
                
            # Extract IPv4 address from this interface block
            inet_match = re.search(r"\n\s+inet (\d+\.\d+\.\d+\.\d+)", iface_block.group(0))
            if not inet_match:
                print(f"     ❌ No IPv4 address found")
                continue
                
            ip = inet_match.group(1)
            print(f"     ✅ Found IP: {ip}")
            
        print("\n4. Evaluating candidates...")
        
        for iface in up_interfaces:
            # Re-extract IP for evaluation
            iface_pattern = re.compile(rf"^{re.escape(iface)}:.*?(?=^[a-zA-Z]|\Z)", re.M | re.S)
            iface_block = iface_pattern.search(ifconfig_out)
            if not iface_block:
                continue
            inet_match = re.search(r"\n\s+inet (\d+\.\d+\.\d+\.\d+)", iface_block.group(0))
            if not inet_match:
                continue
            ip = inet_match.group(1)
            print(f"\n   Evaluating {iface} ({ip}):")
            
            try:
                ip_addr = ipaddress.ip_address(ip)
                
                # Skip loopback
                if ip_addr.is_loopback:
                    print(f"     ❌ Skipping - loopback address")
                    continue
                    
                # Skip carrier-grade NAT 100.64.0.0/10 which is not routable P2P
                if ip_addr >= ipaddress.ip_address("100.64.0.0") and ip_addr <= ipaddress.ip_address("100.127.255.255"):
                    print(f"     ❌ Skipping - carrier-grade NAT")
                    continue
                
                # Test if we can actually bind to this interface
                print(f"     🧪 Testing socket binding...")
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    test_sock.bind((ip, 0))  # bind to any free port
                    port = test_sock.getsockname()[1]
                    test_sock.close()
                    print(f"     ✅ Binding test passed (port {port})")
                    
                    # Calculate priority
                    priority = 0
                    if iface == "en0":  # prefer en0 (usually Wi-Fi)
                        priority = 100
                        print(f"     🎯 Priority 100 (en0 - Wi-Fi interface)")
                    elif ip_addr.is_private:  # prefer private IPs
                        priority = 50
                        print(f"     🎯 Priority 50 (private IP)")
                    elif iface.startswith("en") and int(iface[2:]) < 4:  # prefer en0-en3
                        priority = 25
                        print(f"     🎯 Priority 25 (en0-en3 range)")
                    else:
                        print(f"     🎯 Priority 0 (default)")
                    
                    candidates.append((priority, iface, ip))
                    print(f"     ✅ Added to candidates with priority {priority}")
                    
                except OSError as e:
                    print(f"     ❌ Binding test failed: {e}")
                    continue
                    
            except (ValueError, OSError) as e:
                print(f"     ❌ IP validation failed: {e}")
                continue
        
        print(f"\n5. Final candidate ranking:")
        if candidates:
            candidates.sort(reverse=True)
            for i, (priority, iface, ip) in enumerate(candidates):
                marker = "🏆" if i == 0 else "  "
                print(f"   {marker} {priority}: {iface} ({ip})")
            
            winner = candidates[0]
            print(f"\n6. Selected: {winner[1]} ({winner[2]})")
            return winner[1], winner[2]
        else:
            print("\n6. No candidates found!")
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
    
    # Fallback: try to get en0 IP
    print("\n7. Trying fallback to en0...")
    try:
        ip_result = subprocess.run(["ipconfig", "getifaddr", "en0"],
                                   capture_output=True, text=True, check=True)
        fallback_ip = ip_result.stdout.strip()
        print(f"   ✅ en0 fallback successful: {fallback_ip}")
        return "en0", fallback_ip
    except Exception as e:
        print(f"   ❌ en0 fallback failed: {e}")
        print(f"   Using final fallback: en0, 127.0.0.1")
        return "en0", "127.0.0.1"

if __name__ == "__main__":
    result = debug_pick_macos_iface()
    print(f"\n🎯 Final result: {result[0]} ({result[1]})") 