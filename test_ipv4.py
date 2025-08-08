#!/usr/bin/env python3
"""
Test script to verify IPV4 functionality in Arceus.
This script will:
1. Test getting the local IPV4 address
2. Test interface selection on macOS
3. Test socket binding with explicit IPV4
4. Print diagnostic information
"""

import socket
import platform
import os
import sys
from arceus.networking import get_local_ip, get_broadcast_ip, find_free_port, UDPBeacon
from arceus.utils import setup_macos_gloo_env, validate_gloo_setup, _pick_macos_iface

def test_ipv4_address():
    """Test getting IPV4 address"""
    print("\n==== Testing IPV4 Address Resolution ====")
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    
    # Verify it's a valid IPV4 address
    try:
        socket.inet_pton(socket.AF_INET, local_ip)
        print("✓ Valid IPV4 address")
    except (socket.error, OSError):
        print("✗ Not a valid IPV4 address!")
        return False
    
    broadcast_ip = get_broadcast_ip()
    print(f"Broadcast IP: {broadcast_ip}")
    
    free_port = find_free_port()
    print(f"Free port: {free_port}")
    
    return True

def test_macos_interface():
    """Test macOS interface selection"""
    if platform.system() != "Darwin":
        print("\n==== Skipping macOS interface test (not on macOS) ====")
        return True
    
    print("\n==== Testing macOS Interface Selection ====")
    try:
        iface, ip = _pick_macos_iface()
        print(f"Selected interface: {iface}")
        print(f"Interface IP: {ip}")
        
        # Verify it's a valid IPV4 address
        try:
            socket.inet_pton(socket.AF_INET, ip)
            print("✓ Valid IPV4 address for interface")
        except (socket.error, OSError):
            print("✗ Not a valid IPV4 address for interface!")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error selecting interface: {e}")
        return False

def test_socket_binding():
    """Test socket binding with explicit IPV4"""
    print("\n==== Testing IPV4 Socket Binding ====")
    try:
        # Create a socket with explicit IPV4
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to a free port on all interfaces
        port = find_free_port()
        sock.bind(("0.0.0.0", port))
        sock.listen(1)
        
        print(f"✓ Successfully bound to 0.0.0.0:{port}")
        sock.close()
        return True
    except Exception as e:
        print(f"✗ Socket binding failed: {e}")
        return False

def test_udp_beacon():
    """Test UDP beacon functionality"""
    print("\n==== Testing UDP Beacon ====")
    try:
        beacon = UDPBeacon("TEST", find_free_port())
        print("✓ UDP Beacon created successfully")
        beacon.stop()
        return True
    except Exception as e:
        print(f"✗ UDP Beacon creation failed: {e}")
        return False

def test_gloo_setup():
    """Test Gloo setup for macOS"""
    if platform.system() != "Darwin":
        print("\n==== Skipping Gloo setup test (not on macOS) ====")
        return True
    
    print("\n==== Testing Gloo Setup ====")
    try:
        ipaddr = setup_macos_gloo_env()
        print(f"Gloo setup successful, using IP: {ipaddr}")
        
        # Print environment variables
        print("Gloo environment variables:")
        for key in sorted(os.environ.keys()):
            if key.startswith("GLOO_"):
                print(f"  {key}: {os.environ[key]}")
        
        # Test validation
        if validate_gloo_setup():
            print("✓ Gloo validation passed")
            return True
        else:
            print("✗ Gloo validation failed")
            return False
    except Exception as e:
        print(f"✗ Gloo setup failed: {e}")
        return False

def print_system_info():
    """Print system information"""
    print("\n==== System Information ====")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Node: {platform.node()}")
    
    # Print network interfaces (for macOS)
    if platform.system() == "Darwin":
        try:
            import subprocess
            print("\nNetwork interfaces:")
            ifconfig_out = subprocess.check_output(["ifconfig"]).decode()
            print(ifconfig_out)
        except Exception as e:
            print(f"Failed to get network interfaces: {e}")

if __name__ == "__main__":
    print("===== IPV4 Functionality Test =====")
    
    print_system_info()
    
    tests_passed = 0
    tests_failed = 0
    
    test_functions = [
        test_ipv4_address,
        test_macos_interface,
        test_socket_binding,
        test_udp_beacon,
        test_gloo_setup
    ]
    
    for test_func in test_functions:
        if test_func():
            tests_passed += 1
        else:
            tests_failed += 1
    
    print("\n===== Test Results =====")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✅ All tests passed - IPV4 functionality appears to be working correctly")
        sys.exit(0)
    else:
        print(f"\n❌ {tests_failed} tests failed - please check the logs above for details")
        sys.exit(1)