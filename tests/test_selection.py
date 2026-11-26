"""The display subsample and the screen-space lasso (core/selection.py)."""

import numpy as np

from cloudlabeller.core.selection import (
    even_subsample,
    matrix_from_column_major,
    points_in_polygon,
    project_to_css,
    select_in_polygon,
)


def test_even_subsample_is_seeded_sorted_and_optional():
    a = even_subsample(1000, 100)
    assert len(a) == 100 and (np.diff(a) > 0).all()
    assert np.array_equal(a, even_subsample(1000, 100))
    assert even_subsample(1000, 0) is None and even_subsample(1000, 1000) is None


def test_even_odd_rule():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    pts = np.array([[5, 5], [15, 5], [-1, -1], [9.9, 0.1]])
    assert points_in_polygon(pts, square).tolist() == [True, False, False, True]
    # A pentagram: its centre is circled twice, so even-odd puts it outside.
    angles = np.radians(90 + 144 * np.arange(5))
    star = np.column_stack([np.cos(angles), np.sin(angles)])
    tip = star[0] * 0.8
    assert points_in_polygon(np.array([[0.0, 0.0], tip]), star).tolist() == [False, True]


def test_projection_uses_css_pixels_and_drops_points_behind():
    # Orthographic-like: clip = (x, y, 0, w) with w = z; screen = ndc mapped.
    m = np.zeros((4, 4))
    m[0, 0] = m[1, 1] = 1.0
    m[3, 2] = 1.0                                   # w = z
    assert np.array_equal(matrix_from_column_major(m.T.reshape(16)), m)
    xyz = np.array([[0.0, 0.0, 1.0], [-0.9, 0.9, 1.0], [0.5, 0.0, -1.0]])
    screen, in_front = project_to_css(xyz, m, 200, 100)
    assert np.allclose(screen[0], [100, 50])        # centre
    assert np.allclose(screen[1], [10, 5])          # near the top-left corner
    assert in_front.tolist() == [True, True, False]

    # The point behind the camera projects to (-0.5 ndc), inside the lasso
    # drawn over the whole screen, and is still not selected.
    everything = [[0, 0], [200, 0], [200, 100], [0, 100]]
    picked = select_in_polygon(xyz, m.T.reshape(16), everything, 200, 100)
    assert picked.dtype == np.uint32 and picked.tolist() == [0, 1]
