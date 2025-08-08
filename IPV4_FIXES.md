# IPV4 Fixes for Cross-Macbook Communication

This document describes the changes made to ensure consistent IPV4 usage for cross-Macbook communication in the Arceus distributed training framework.

## Summary of Changes

1. **Explicit IPV4 Usage**
   - Modified socket creation to explicitly use `socket.AF_INET` 
   - Changed empty string binding ("") to explicit "0.0.0.0" for IPV4 interfaces
   - Added IPV4 validation using `socket.inet_pton(socket.AF_INET, ip)`
   - Improved fallback mechanisms when IPV4 resolution fails

2. **macOS-specific Improvements**
   - Enhanced `_pick_macos_iface()` to prioritize stable IPV4 interfaces
   - Improved handling of virtual/Docker interfaces that might cause connectivity issues
   - Added better error handling and logging for interface selection
   - Set environment variables to force IPV4 usage with Gloo backend

3. **Environment Variable Configuration**
   - Set `GLOO_SOCKET_FAMILY = "AF_INET"` to force IPV4 socket family
   - Set `GLOO_SOCKET_DISABLE_IPV6 = "1"` to prevent IPV6 fallback
   - Properly propagate IP addresses between hosts/joiners with explicit IPV4

4. **Distributed Communication**
   - Added IPV4 validation and resolution in TrainingJoiner
   - Updated TrainingHost to bind to all IPV4 interfaces
   - Improved IP address propagation between hosts and joiners

## Verification

The IPV4 functionality was tested with:
- Socket creation and binding tests
- Hostname resolution to IPV4
- Interface detection and binding on macOS
- Cross-Macbook communication simulation

All tests pass on macOS, showing proper IPV4 usage throughout the system.

## Usage Notes

When using the distributed training functionality:
1. The system now automatically sets up the macOS environment for IPV4
2. The `train.py` script calls `arceus.setup_macos_env()` on macOS systems
3. Interface selection will prioritize stable Wi-Fi connections (typically en0)
4. All socket communications use explicit IPV4

## Future Improvements

1. Add support for manual IPV4 address specification via environment variables
2. Improve error messages for network connectivity issues
3. Add more robust detection of problematic network configurations