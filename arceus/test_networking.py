import pytest
from arceus.networking import UDPBeacon, find_free_port


def test_udp_beacon_initialization():
    session_id = "test_session"
    tcp_port = find_free_port()
    beacon = UDPBeacon(session_id, tcp_port)
    
    assert beacon.session_id == session_id
    assert beacon.tcp_port == tcp_port
    assert beacon.running is True
    
    beacon.stop()


def test_udp_beacon_communication():
    session_id = "test_session"
    tcp_port = find_free_port()
    beacon = UDPBeacon(session_id, tcp_port)
    
    # Allow some time for potential communication
    import time
    time.sleep(2)

    # Check that no active sessions are found as this is a standalone test
    active_sessions = beacon.get_active_sessions()
    assert session_id not in active_sessions

    beacon.stop()