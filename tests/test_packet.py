from ballistic.common.packet import PACKET_SIZE, Telemetry


def test_packet_round_trip():
    t = Telemetry(
        t_sim=12.34,
        lat=1.3521, lon=103.8198, alt=842.5,
        q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0,
        v_n=10.0, v_e=20.0, v_d=-5.0,
        p=0.1, q=0.2, r=0.3,
        phase=2, round_id=2,
    )
    raw = t.pack()
    assert len(raw) == PACKET_SIZE
    t2 = Telemetry.unpack(raw)
    assert t2.t_sim == t.t_sim
    assert t2.lat == t.lat
    assert t2.lon == t.lon
    assert t2.phase == t.phase
    assert t2.round_id == t.round_id
    # f32 quaternion fields tolerate small roundoff
    assert abs(t2.q_w - t.q_w) < 1e-6
