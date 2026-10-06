import numpy as np
from scipy.integrate import trapezoid
from ssme_lite.reporting.data import bounded_density


def test_reflected_density_and_point_mass():
    for values, lo, hi in [(np.r_[np.ones(998), .999, .999], .5, 1),
                           (np.r_[np.zeros(998), .008, .008], 0, np.log(2))]:
        d = bounded_density(values, lo, hi)
        assert min(d['x']) >= lo and max(d['x']) <= hi
        assert np.isfinite(d['density']).all()
        assert min(d['density']) >= 0
        assert abs(trapezoid(d['density'], d['x'])-1) < .002
    assert bounded_density(np.ones(20), .5, 1)['point_mass'] == 1
    assert bounded_density([], 0, 1)['n'] == 0
