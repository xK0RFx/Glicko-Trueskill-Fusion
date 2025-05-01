import pytest
from src.glicko2 import _g, _E, _v, _delta


def test_g_zero_phi():
    assert pytest.approx(_g(0.0), rel=1e-9) == 1.0


def test_g_positive_phi():
    assert _g(1.0) < 1.0
    assert _g(0.5) > _g(1.0)


def test_E_equal_mus():
    for phi in [0.1, 1.0, 2.0]:
        val = _E(0.0, 0.0, phi)
        assert pytest.approx(val, rel=1e-6) == 0.5


def test_v_positive():
    val = _v(0.0, 0.0, 1.0)
    assert val > 0.0


def test_delta_sign():
    # Outcome better than expected => positive delta
    v = _v(0.0, 1.0, 1.0)
    delta = _delta(0.0, 1.0, 1.0, v, 1.0)
    assert delta > 0
    # Outcome worse => negative
    delta2 = _delta(0.0, 1.0, 1.0, v, 0.0)
    assert delta2 < 0 