#!/usr/bin/env python3
"""
Test script to verify interface selection works correctly on both machines.
Run this on both the host and joining machine before attempting distributed training.
"""

import os
import sys
import arceus

def test_interface_selection():
    """Test that interface selection works correctly"""
    print("🧪 Testing Arceus Interface Selection")
    print("=" * 50)
    
    # Test 1: Check what the diagnostic script recommends
    print("\n1. Running diagnostic script...")
    try:
        os.system("python debug_interfaces.py")
    except Exception as e:
        print(f"   ❌ Diagnostic script failed: {e}")
        return False
    
    # Test 2: Test Arceus setup
    print("\n2. Testing Arceus setup...")
    try:
        ip = arceus.setup_macos_env()
        print(f"   ✅ Arceus selected IP: {ip}")
        
        # Show what environment variables were set
        print(f"   GLOO_SOCKET_IFNAME: {os.getenv('GLOO_SOCKET_IFNAME')}")
        print(f"   GLOO_SOCKET_IFADDR: {os.getenv('GLOO_SOCKET_IFADDR')}")
        
    except Exception as e:
        print(f"   ❌ Arceus setup failed: {e}")
        return False
    
    # Test 3: Test Gloo validation
    print("\n3. Testing Gloo validation...")
    try:
        arceus.validate_gloo()
        print("   ✅ Gloo validation passed")
    except Exception as e:
        print(f"   ❌ Gloo validation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! This machine is ready for distributed training.")
    return True

if __name__ == "__main__":
    success = test_interface_selection()
    sys.exit(0 if success else 1) 