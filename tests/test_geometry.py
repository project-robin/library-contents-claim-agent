import numpy as np

from pipeline.geometry import floor_ceiling, room_polygon, room_shape, shoelace

rng = np.random.default_rng(1)


def box_room(L=4.2, W=3.1, H=2.7, n=40000, angle=17):
    """Points on floor, ceiling and 4 walls of a rotated box room, plus a bookcase and table inside."""
    pts = []
    for _ in range(n):
        k = rng.integers(6)
        x, y, z = rng.uniform(0, L), rng.uniform(0, W), rng.uniform(0, H)
        pts.append([(x, y, 0), (x, y, H), (0, y, z), (L, y, z), (x, 0, z), (x, W, z)][k])
    pts = np.array(pts, float)
    shelf = np.c_[rng.uniform(0.5, 2.0, 4000), np.full(4000, 0.3), rng.uniform(0, 2.0, 4000)]  # 30 cm off wall
    table = np.c_[rng.uniform(1.5, 2.5, 3000), rng.uniform(1.0, 2.0, 3000), np.full(3000, 0.75)]
    pts = np.r_[pts, shelf, table] + rng.normal(0, 0.01, (len(pts) + 7000, 3))
    a = np.radians(angle)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return pts @ R.T


def test_rectangle_room_dims():
    pts = box_room()
    up, fz, cz = floor_ceiling(pts, np.array([0.05, -0.03, 1.0]) / np.linalg.norm([0.05, -0.03, 1.0]))
    assert abs(fz - 0) < 0.03 and abs(cz - fz - 2.7) < 0.05
    h = pts @ up - fz
    wall = pts[(h > 0.3) & (h < 2.4)]
    e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    rot, poly, note = room_shape(np.c_[wall @ e1, wall @ e2])
    assert "rectangle" in note
    lw = sorted([np.ptp(poly[:, 0]), np.ptp(poly[:, 1])], reverse=True)
    assert abs(lw[0] - 4.2) / 4.2 < 0.03 and abs(lw[1] - 3.1) / 3.1 < 0.03, lw  # bookcase front ignored
    assert abs(shoelace(poly) - 4.2 * 3.1) / (4.2 * 3.1) < 0.05


def test_l_shaped_room_is_polygon():
    # L = 4x4 square minus a 2x2 corner -> area 12
    segs = [((0, 0), (4, 0)), ((4, 0), (4, 2)), ((4, 2), (2, 2)), ((2, 2), (2, 4)), ((2, 4), (0, 4)), ((0, 4), (0, 0))]
    xy = np.concatenate([np.array(a) + rng.uniform(0, 1, (3000, 1)) * (np.array(b) - np.array(a)) for a, b in segs])
    xy += rng.normal(0, 0.01, xy.shape)
    rot, poly, note = room_shape(xy)
    assert "polygon" in note, note
    assert abs(shoelace(poly) - 12) / 12 < 0.10


def test_open_ring_rejected():
    t = rng.uniform(0, 1.5 * np.pi, 5000)  # three quarters of a wall ring
    xy = np.c_[2 * np.cos(t), 2 * np.sin(t)]
    assert room_polygon(xy) is None
