#!/usr/bin/env python3
"""
Simple IPV4 test script that doesn't require dependencies.
This script will:
1. Test socket creation with explicit IPV4
2. Test hostname resolution to IPV4
3. Test interface detection and binding
"""

import socket
import platform
import os
import sys
import re
import subprocess
from contextlib import closing

def get_local_ip():
    """Get local IPV4 address using Google DNS"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        # Verify this is a valid IPV4 address
        socket.inet_pton(socket.AF_INET, ip)  # Will raise an error if not valid IPV4
        return ip
    except (socket.error, OSError):
        # Fallback to using the hostname resolution (should be IPV4)
        try:
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)  # Returns first (and usually only) IPV4 address
        except socket.error:
            # Last resort fallback
            return "127.0.0.1"
    finally:
        sock.close()

def find_free_port():
    """Find a free port, explicitly using IPV4"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("0.0.0.0", 0))  # Explicitly bind to all available IPV4 interfaces
        return s.getsockname()[1]

def pick_macos_iface():
    """Find best macOS interface for IPV4 communication"""
    if platform.system() != "Darwin":
        return "eth0", "127.0.0.1"  # sensible default on non-macOS
    
    try:
        ifconfig_out = subprocess.check_output(["ifconfig"]).decode()
        
        # Find all UP interfaces
        up_interfaces = re.findall(r"^(en\d+):.*?<UP,.*?>", ifconfig_out, re.M)
        
        for iface in up_interfaces:
            # For each UP interface, find its IPv4 address
            # Look for the interface block and extract the inet address
            iface_pattern = re.compile(rf"^{re.escape(iface)}:.*?(?=^[a-zA-Z]|\Z)", re.M | re.S)
            iface_block = iface_pattern.search(ifconfig_out)
            
            if not iface_block:
                continue
                
            # Extract IPv4 address from this interface block
            inet_match = re.search(r"\n\s+inet (\d+\.\d+\.\d+\.\d+)", iface_block.group(0))
            if not inet_match:
                continue
                
            ip = inet_match.group(1)
            try:
                # Verify it's a valid IPV4 address
                socket.inet_pton(socket.AF_INET, ip)
                
                # Skip loopback
                if ip == "127.0.0.1":
                    continue
                
                # Test if we can actually bind to this interface
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    test_sock.bind((ip, 0))  # bind to any free port
                    test_sock.close()
                    
                    return iface, ip
                    
                except OSError:
                    # Can't bind to this interface, skip it
                    continue
                    
            except (ValueError, OSError):
                continue
                
    except Exception as e:
        print(f"Error finding interface: {e}")
    
    # Fallback to en0
    try:
        ip_result = subprocess.run(["ipconfig", "getifaddr", "en0"],
                                  capture_output=True, text=True, check=True)
        return "en0", ip_result.stdout.strip()
    except Exception as e:
        print(f"Fallback to en0 failed: {e}")
        return "lo0", "127.0.0.1"

def test_ipv4_socket():
    """Test IPV4 socket creation and binding"""
    print("\n=== Testing IPV4 Socket Creation ===")
    try:
        # Create socket with explicit IPV4
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Get a free port
        port = find_free_port()
        print(f"Found free port: {port}")
        
        # Try binding to all interfaces
        sock.bind(("0.0.0.0", port))
        print("✓ Successfully bound to all IPV4 interfaces (0.0.0.0)")
        sock.close()
        
        # Try binding to localhost
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        print("✓ Successfully bound to localhost (127.0.0.1)")
        sock.close()
        
        # Try binding to specific interface
        local_ip = get_local_ip()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((local_ip, port))
        print(f"✓ Successfully bound to specific interface ({local_ip})")
        sock.close()
        
        return True
    except Exception as e:
        print(f"✗ Socket test failed: {e}")
        return False

def test_hostname_resolution():
    """Test hostname resolution to IPV4"""
    print("\n=== Testing Hostname Resolution ===")
    
    try:
        hostname = socket.gethostname()
        print(f"Hostname: {hostname}")
        
        # Resolve hostname to address
        try:
            ip = socket.gethostbyname(hostname)
            print(f"Resolved to: {ip}")
            
            # Verify it's IPV4
            socket.inet_pton(socket.AF_INET, ip)
            print(f"✓ Successfully resolved hostname to IPV4 address")
            return True
        except socket.error as e:
            print(f"✗ Failed to resolve hostname to IPV4: {e}")
            return False
    except Exception as e:
        print(f"✗ Hostname resolution test failed: {e}")
        return False

def test_interface_detection():
    """Test interface detection and binding"""
    print("\n=== Testing Interface Detection ===")
    
    try:
        if platform.system() == "Darwin":
            iface, ip = pick_macos_iface()
            print(f"Selected macOS interface: {iface} with IP: {ip}")
            
            # Try binding to this interface
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((ip, 0))
                bound_port = sock.getsockname()[1]
                print(f"✓ Successfully bound to interface {iface} ({ip}) on port {bound_port}")
                sock.close()
                return True
            except Exception as e:
                print(f"✗ Failed to bind to interface {iface}: {e}")
                return False
        else:
            print("Not running on macOS, skipping macOS-specific interface detection")
            return True
    except Exception as e:
        print(f"✗ Interface detection test failed: {e}")
        return False

def print_system_info():
    """Print system information"""
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Node: {platform.node()}")
    
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    
    # Print all local addresses
    print("\nAll local addresses:")
    try:
        hostname = socket.gethostname()
        addrs = socket.getaddrinfo(hostname, None)
        for addr in addrs:
            family, socktype, proto, canonname, sockaddr = addr
            if family == socket.AF_INET:  # IPV4
                print(f"  IPV4: {sockaddr[0]}")
            elif family == socket.AF_INET6:  # IPV6
                print(f"  IPV6: {sockaddr[0]}")
    except Exception as e:
        print(f"  Error getting addresses: {e}")

if __name__ == "__main__":
    print("===== Simple IPV4 Functionality Test =====")
    
    print_system_info()
    
    tests = [
        test_ipv4_socket,
        test_hostname_resolution,
        test_interface_detection
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n===== Test Results =====")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("\n✅ All tests passed - IPV4 functionality appears to be working correctly")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} tests failed - please check the logs above for details")
        sys.exit(1)