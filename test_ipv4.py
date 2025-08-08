#!/usr/bin/env python3
"""
Simple test script for validating IPv4 configuration.
"""

import socket
import platform
import subprocess
import re

def print_section(title):
    """Print a section header for better readability."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def get_local_ip():
    """Get the local IP address that can be used for network communication.
    
    This function explicitly uses IPv4 and tries to find a suitable interface
    by connecting to an external service.
    """
    # Always use IPv4 (AF_INET)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to Google DNS to determine which interface to use
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        # Fallback - try to get a non-loopback IPv4 address
        try:
            # Get all IPv4 addresses on all interfaces
            hostname = socket.gethostname()
            for ip in socket.getaddrinfo(hostname, None, socket.AF_INET):
                # Skip loopback addresses (127.x.x.x)
                if not ip[4][0].startswith('127.'):
                    return ip[4][0]
        except Exception:
            pass
        # Last resort fallback
        return "127.0.0.1"
    finally:
        sock.close()

def get_broadcast_ip(local_ip):
    """Get the broadcast IP address for the local network."""
    # Get the local IP and replace last octet with 255
    parts = local_ip.split(".")
    parts[3] = "255"  
    return ".".join(parts)

def find_free_port():
    """Find a free port on the local machine."""
    # Explicitly use IPv4 (AF_INET)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def test_socket_creation():
    """Test socket creation with explicit IPv4."""
    print_section("Testing IPv4 Socket Creation")
    
    try:
        # Create an explicit IPv4 socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("✓ Created IPv4 socket successfully")
        
        # Try to bind to localhost
        try:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            print(f"✓ Successfully bound to 127.0.0.1:{port}")
        except Exception as e:
            print(f"✗ Error binding to localhost: {e}")
        
        # Try to bind to the detected local IP
        local_ip = get_local_ip()
        try:
            sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock2.bind((local_ip, 0))
            port = sock2.getsockname()[1]
            print(f"✓ Successfully bound to {local_ip}:{port}")
            sock2.close()
        except Exception as e:
            print(f"✗ Error binding to {local_ip}: {e}")
        
        sock.close()
    except Exception as e:
        print(f"✗ Error creating socket: {e}")

def test_network_detection():
    """Test the network interface and IP detection logic."""
    print_section("Testing Network Interface Detection")
    
    try:
        local_ip = get_local_ip()
        broadcast_ip = get_broadcast_ip(local_ip)
        free_port = find_free_port()
        
        print(f"Local IP: {local_ip}")
        print(f"Broadcast IP: {broadcast_ip}")
        print(f"Free port: {free_port}")
        
        # Validate IPv4 format
        if "." in local_ip and len(local_ip.split(".")) == 4:
            print("✓ Local IP is in valid IPv4 format")
        else:
            print("✗ Local IP is NOT in valid IPv4 format")
            
        if "." in broadcast_ip and len(broadcast_ip.split(".")) == 4:
            print("✓ Broadcast IP is in valid IPv4 format")
        else:
            print("✗ Broadcast IP is NOT in valid IPv4 format")
        
        # On macOS, check interfaces with ifconfig
        if platform.system() == "Darwin":
            try:
                print("\nDetected network interfaces (macOS):")
                ifconfig_output = subprocess.check_output(["ifconfig"]).decode()
                
                # Find all interfaces and their IPv4 addresses
                interfaces = re.findall(r"^([a-zA-Z0-9]+):.*?(?=^[a-zA-Z0-9]|\Z)", ifconfig_output, re.M | re.S)
                
                for iface in interfaces:
                    # Extract the interface block
                    iface_pattern = re.compile(rf"^{re.escape(iface)}:.*?(?=^[a-zA-Z0-9]|\Z)", re.M | re.S)
                    iface_block = iface_pattern.search(ifconfig_output)
                    
                    if iface_block:
                        # Extract IPv4 address
                        inet_match = re.search(r"\n\s+inet (\d+\.\d+\.\d+\.\d+)", iface_block.group(0))
                        ip = inet_match.group(1) if inet_match else "No IPv4"
                        
                        # Check if this interface is UP
                        is_up = "UP" in iface_block.group(0)
                        up_status = "UP" if is_up else "DOWN"
                        
                        print(f"  {iface}: {ip} ({up_status})")
                        
                        # Check if this is the interface with our detected IP
                        if ip == local_ip:
                            print(f"  ↳ This is the interface being used for networking")
            except Exception as e:
                print(f"Error getting interface details: {e}")
    except Exception as e:
        print(f"Error in network detection: {e}")

def main():
    """Run all tests."""
    print_section("IPv4 Validation Tests")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    test_network_detection()
    test_socket_creation()
    
    print("\nAll tests completed. If you see any failures, please address them.")

if __name__ == "__main__":
    main()