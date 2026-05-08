import numpy as np

from ballistic.sim.aero import CD_FIN, CD_SPIN, cd_lookup


def test_cd_endpoints_clamp():
    assert cd_lookup(-1.0, CD_FIN) == CD_FIN[0, 1]
    assert cd_lookup(99.0, CD_FIN) == CD_FIN[-1, 1]


def test_cd_transonic_peaks_above_subsonic():
    sub = cd_lookup(0.5, CD_SPIN)
    transonic = cd_lookup(1.0, CD_SPIN)
    supersonic = cd_lookup(3.0, CD_SPIN)
    assert transonic > sub
    assert transonic > supersonic
