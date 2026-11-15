"""lr_schedule "wsd" (2026-11-15): linear warmup, constant 1, linear decay to the min scale at n_epochs."""
import pytest


def test_wsd_schedule_shape():
    from bam_omat24.train_3bpa import wsd_schedule
    fn = wsd_schedule(100, 800, 1000, 0.05)
    f = lambda t: float(fn(t))  # noqa: E731
    assert abs(f(0) - 0.05) < 1e-6 and abs(f(50) - 0.525) < 1e-6 and abs(f(100) - 1.0) < 1e-6
    assert f(400) == 1.0 and f(799) == 1.0
    assert abs(f(900) - 0.525) < 1e-6 and abs(f(1000) - 0.05) < 1e-6 and abs(f(1500) - 0.05) < 1e-6
    with pytest.raises(ValueError):
        wsd_schedule(100, 1000, 1000, 0.05)
