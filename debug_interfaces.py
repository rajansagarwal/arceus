#!/usr/bin/env python3
"""
Comprehensive macOS Network Interface Diagnostic Script
Helps debug PyTorch Gloo interface binding issues
"""

import subprocess
import socket
import re
import ipaddress
import platform
import os

def get_all_interfaces():
    """Get all network interfaces and their details"""
    try:
        ifconfig_out = subprocess.check_output(["ifconfig"]).decode()
        
        # Parse interfaces
        interfaces = {}
        current_iface = None
        
        for line in ifconfig_out.split('\n'):
            # Interface line (starts with interface name)
            if re.match(r'^[a-zA-Z0-9]+:', line):
                current_iface = line.split(':')[0]
                interfaces[current_iface] = {
                    'flags': line,
                    'ipv4': None,
                    'ipv6': [],
                    'status': 'unknown',
                    'can_bind': False
                }
                
                # Extract flags
                if '<UP,' in line:
                    interfaces[current_iface]['status'] = 'up'
                else:
                    interfaces[current_iface]['status'] = 'down'
                    
            elif current_iface and line.strip():
                # IPv4 address
                if 'inet ' in line and 'inet6' not in line:
                    match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
                    if match:
                        interfaces[current_iface]['ipv4'] = match.group(1)
                        
                # IPv6 addresses
                elif 'inet6' in line:
                    match = re.search(r'inet6 ([a-fA-F0-9:]+)', line)
                    if match:
                        interfaces[current_iface]['ipv6'].append(match.group(1))
        
        return interfaces
        
    except Exception as e:
        print(f"Error getting interfaces: {e}")
        return {}

def test_bind_capability(ip_addr):
    """Test if we can bind to a given IP address"""
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind((ip_addr, 0))
        port = test_sock.getsockname()[1]
        test_sock.close()
        return True, port
    except Exception as e:
        return False, str(e)

def categorize_ip(ip_str):
    """Categorize an IP address"""
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_loopback:
            return "loopback"
        elif ip.is_private:
            if ip >= ipaddress.ip_address("100.64.0.0") and ip <= ipaddress.ip_address("100.127.255.255"):
                return "carrier-grade-nat"
            else:
                return "private"
        elif ip.is_link_local:
            return "link-local"
        else:
            return "public"
    except:
        return "unknown"

def main():
    print("🔍 macOS Network Interface Diagnostic Tool")
    print("=" * 50)
    
    if platform.system() != "Darwin":
        print("❌ This script is designed for macOS only")
        return
    
    # Get all interfaces
    interfaces = get_all_interfaces()
    
    print(f"\n📡 Found {len(interfaces)} network interfaces:")
    print()
    
    suitable_interfaces = []
    
    for iface, details in interfaces.items():
        print(f"Interface: {iface}")
        print(f"  Status: {details['status']}")
        print(f"  Flags: {details['flags']}")
        
        if details['ipv4']:
            ip_category = categorize_ip(details['ipv4'])
            can_bind, bind_info = test_bind_capability(details['ipv4'])
            
            print(f"  IPv4: {details['ipv4']} ({ip_category})")
            print(f"  Can bind: {'✓' if can_bind else '✗'} {bind_info if not can_bind else f'(tested port {bind_info})'}")
            
            # Check if suitable for Gloo
            if (details['status'] == 'up' and 
                details['ipv4'] and 
                ip_category in ['private', 'public'] and 
                can_bind):
                
                priority = 0
                if iface == "en0":
                    priority = 100
                elif ip_category == "private":
                    priority = 50
                elif iface.startswith("en") and iface[2:].isdigit() and int(iface[2:]) < 4:
                    priority = 25
                
                suitable_interfaces.append((priority, iface, details['ipv4'], ip_category))
                print(f"  ✅ Suitable for Gloo (priority: {priority})")
            else:
                print(f"  ❌ Not suitable for Gloo")
        else:
            print(f"  IPv4: None")
            print(f"  ❌ No IPv4 address")
        
        if details['ipv6']:
            print(f"  IPv6: {', '.join(details['ipv6'])}")
        
        print()
    
    # Show recommendation
    print("🎯 Gloo Interface Recommendation:")
    print("-" * 30)
    
    if suitable_interfaces:
        suitable_interfaces.sort(reverse=True)  # Sort by priority
        best_iface = suitable_interfaces[0]
        
        print(f"✅ Best interface: {best_iface[1]} ({best_iface[2]})")
        print(f"   Category: {best_iface[3]}")
        print(f"   Priority: {best_iface[0]}")
        
        if len(suitable_interfaces) > 1:
            print(f"\n📋 Other suitable interfaces:")
            for priority, iface, ip, category in suitable_interfaces[1:]:
                print(f"   {iface} ({ip}) - {category} (priority: {priority})")
    else:
        print("❌ No suitable interfaces found!")
        print("   This explains why Gloo is failing.")
    
    # Show current environment
    print(f"\n🔧 Current Gloo Environment:")
    print("-" * 30)
    gloo_vars = [
        "GLOO_SOCKET_IFNAME",
        "GLOO_SOCKET_IFADDR", 
        "GLOO_SOCKET_FAMILY",
        "GLOO_SOCKET_DISABLE_IPV6",
        "GLOO_ALLOW_UNSECURED"
    ]
    
    for var in gloo_vars:
        value = os.environ.get(var, "not set")
        print(f"  {var}: {value}")
    
    # Test arceus setup
    print(f"\n🧪 Testing Arceus Setup:")
    print("-" * 30)
    
    try:
        import arceus
        result = arceus.setup_macos_env()
        print(f"✅ Arceus setup completed, returned IP: {result}")
    except Exception as e:
        print(f"❌ Arceus setup failed: {e}")

if __name__ == "__main__":
    main() 