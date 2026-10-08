"""Metric geometry: marker scale, floor/ceiling, room shape, and 2D box -> 3D measurements.

World = VGGT frame (first keyframe camera), unscaled. Room frame = metres, z up, floor at z=0, x/y along the walls.
"""
import json

import cv2
import numpy as np

from pipeline import settings

ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
INSET = 0.7  # sample marker depth on a square shrunk to 70% so samples stay on the paper


def recon_size(w, h):
    """VGGT input size: long side 518, both sides multiples of 14, no crop (keeps pixel mapping linear)."""
    s = 518 / max(w, h)
    return max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)


def box_px(box_2d, w, h):
    ymin, xmin, ymax, xmax = box_2d
    return np.array([xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000])


def plane_normal(pts):
    return np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][-1]


class Recon:
    def __init__(self, sweep_dir):
        z = np.load(sweep_dir / "recon.npz")
        self.frames = [str(f) for f in z["frames"]]
        self.idx = {f: i for i, f in enumerate(self.frames)}
        self.E, self.K, self.orig = z["extrinsic"], z["intrinsic"], z["orig_wh"]
        self.depth, self.conf = z["depth"].astype(np.float32), z["conf"].astype(np.float32)
        self.gh, self.gw = self.depth.shape[1:3]
        self.metric = bool(z["metric"]) if "metric" in z.files else False  # MapAnything outputs metres
        self.conf_thr = np.median(self.conf.reshape(len(self.frames), -1), axis=1)

    def to_grid(self, i, uv):
        w, h = self.orig[i]
        return np.asarray(uv, float) * [self.gw / w, self.gh / h]

    def unproject(self, i, uv_grid, d):
        cam = (np.c_[uv_grid, np.ones(len(uv_grid))] @ np.linalg.inv(self.K[i]).T) * np.asarray(d)[:, None]
        R, t = self.E[i][:, :3], self.E[i][:, 3]
        return (cam - t) @ R  # R^T (cam - t)

    def depth_at(self, i, p):
        x, y = int(round(p[0])), int(round(p[1]))
        d = self.depth[i, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        c = self.conf[i, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        ok = d[c >= self.conf_thr[i]]
        return float(np.median(ok)) if ok.size else None

    def _box_slice(self, i, bpx):
        g = self.to_grid(i, bpx.reshape(2, 2)).ravel()
        x0, y0 = np.floor(g[:2]).astype(int).clip(0)
        x1, y1 = np.ceil(g[2:]).astype(int)
        return x0, y0, min(x1, self.gw), min(y1, self.gh)

    def box_depth(self, i, bpx):
        x0, y0, x1, y1 = self._box_slice(i, bpx)
        d, c = self.depth[i, y0:y1, x0:x1], self.conf[i, y0:y1, x0:x1]
        ok = d[c >= self.conf_thr[i]]
        return float(np.median(ok)) if ok.size >= 3 else None

    def pixels_world(self, i, x0, y0, x1, y1, stride=1):
        ys, xs = np.mgrid[y0:y1:stride, x0:x1:stride]
        d, c = self.depth[i, ys, xs], self.conf[i, ys, xs]
        keep = c >= self.conf_thr[i]
        return self.unproject(i, np.c_[xs[keep], ys[keep]] + 0.5, d[keep])

    def box_points(self, i, bpx):
        return self.pixels_world(i, *self._box_slice(i, bpx))

    def frame_points(self, i, stride=4):
        return self.pixels_world(i, 0, 0, self.gw, self.gh, stride)


# ---------- metric scale ----------

def marker_scale_samples(sweep_dir, rec):
    """Each detected marker gives one scale sample: real side length / reconstructed side length."""
    det = cv2.aruco.ArucoDetector(ARUCO, cv2.aruco.DetectorParameters())
    samples = []
    for i, f in enumerate(rec.frames):
        img = cv2.imread(str(sweep_dir / "frames" / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = det.detectMarkers(img)
        for c in corners if ids is not None else []:
            c = c.reshape(4, 2)
            inset = c.mean(0) + INSET * (c - c.mean(0))
            g = rec.to_grid(i, inset)
            d = [rec.depth_at(i, p) for p in g]
            if None in d:
                continue
            P = rec.unproject(i, g, d)
            edges = np.linalg.norm(P - np.roll(P, 1, 0), axis=1)
            if edges.max() / max(edges.min(), 1e-9) > 1.5:  # depth bled off the paper
                continue
            samples.append({"frame": f, "scale": settings.MARKER_SIZE_CM / 100 / (edges.mean() / INSET)})
    return samples


def spine_raw(rec, i, bpx, orientation):
    """Spine (height, thickness, center, bottom) in unscaled world units, or None without depth."""
    d = rec.box_depth(i, bpx)
    if d is None:
        return None
    x0, y0, x1, y1 = bpx
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    top, bot, left, right, ctr = rec.unproject(i, rec.to_grid(i, [[cx, y0], [cx, y1], [x0, cy], [x1, cy], [cx, cy]]),
                                               [d] * 5)
    vert, horiz = np.linalg.norm(top - bot), np.linalg.norm(left - right)
    height, thick = (vert, horiz) if orientation == "upright" else (horiz, vert)
    return height, thick, ctr, bot


def book_prior_scale(rec, reads):
    """Fallback scale from standard book formats: median upright spine assumed BOOK_PRIOR_HEIGHT_CM."""
    hs = []
    for f, r in reads.items():
        if f not in rec.idx:
            continue
        i = rec.idx[f]
        w, h = rec.orig[i]
        for s in r["spines"]:
            if s["orientation"] == "upright":
                m = spine_raw(rec, i, box_px(s["box_2d"], w, h), "upright")
                if m:
                    hs.append(m[0])
    return (settings.BOOK_PRIOR_HEIGHT_CM / 100 / float(np.median(hs)), len(hs)) if len(hs) >= 10 else (None, len(hs))


# ---------- room shape ----------

def _peaks(values, bin_m, min_frac):
    lo, hi = np.percentile(values, [0.5, 99.5])
    lo, hi = lo - 0.1 * (hi - lo) - 5 * bin_m, hi + 0.1 * (hi - lo) + 5 * bin_m  # outermost walls sit at the extremes
    hist, edges = np.histogram(values[(values >= lo) & (values <= hi)], bins=np.arange(lo, hi + bin_m, bin_m))
    hs = np.convolve(hist, [1, 2, 3, 2, 1], "same") / 9
    thr = min_frac * len(values)
    return [(edges[k] + bin_m / 2, hs[k]) for k in range(1, len(hs) - 1)
            if hs[k] >= hs[k - 1] and hs[k] >= hs[k + 1] and hs[k] >= thr]


def floor_ceiling(pts, up):
    """Floor = lowest dense horizontal layer; refined twice by plane fit. Ceiling = highest layer >= 2 m above."""
    floor_z = None
    for _ in range(3):
        h = pts @ up
        peaks = _peaks(h, 0.02, 0.005)
        if not peaks:
            return up, None, None
        floor_z = peaks[0][0]
        band = pts[np.abs(h - floor_z) < 0.04]
        if len(band) < 50:
            return up, None, None
        n = plane_normal(band)
        n = n if n @ up > 0 else -n
        if n @ up > np.cos(np.radians(20)):
            up = n / np.linalg.norm(n)
    h = pts @ up
    floor_z = float(np.median(h[np.abs(h - floor_z) < 0.04]))
    ceil = [p for p, _ in _peaks(h, 0.02, 0.002) if p - floor_z >= 2.0]
    ceiling_z = float(np.median(h[np.abs(h - ceil[-1]) < 0.04])) if ceil else None
    return up, floor_z, ceiling_z


def manhattan_box(xy):
    """Rectangle room: pick the rotation that makes walls sharpest, then the outermost wall peak on each side.
    Outermost (not strongest) so a bookcase front never stands in for the wall behind it."""
    sub = xy[np.random.default_rng(0).choice(len(xy), min(len(xy), 60000), replace=False)]
    best = None
    for deg in np.arange(0, 90, 1.0):
        a = np.radians(deg)
        r = sub @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        ent = 0
        for k in range(2):
            p = np.histogram(r[:, k], bins=200)[0] + 1e-9
            p = p / p.sum()
            ent -= (p * np.log(p)).sum()
        if best is None or ent < best[0]:
            best = (ent, a)
    a = best[1]
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    r = xy @ rot
    ext = []
    for k in range(2):
        pk = _peaks(r[:, k], 0.02, 0.0)
        top = max(c for _, c in pk)
        walls = [p for p, c in pk if c >= 0.15 * top]
        ext.append((min(walls), max(walls)))
    (x0, x1), (y0, y1) = ext
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ rot.T
    return rot, corners


def room_polygon(xy, cell=0.05, min_count=3):
    """Any-shape room from wall occupancy: closed outer contour, simplified. None if the walls have big gaps."""
    pad = 10
    lo = xy.min(0) - pad * cell
    idx = ((xy - lo) / cell).astype(int)
    grid = np.zeros(idx.max(0)[::-1] + pad + 1, np.int32)
    np.add.at(grid, (idx[:, 1], idx[:, 0]), 1)
    occ = ((grid >= min_count) * 255).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(occ, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    outer = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(outer) < 0.7 * cv2.contourArea(cv2.convexHull(outer)):
        return None  # C-shaped ring: a wall was never seen
    # ponytail: outer edge of the wall band; overestimates by ~half a cell per side, rectangle path is the precise one
    poly = cv2.approxPolyDP(outer, 0.10 / cell, True).reshape(-1, 2)
    return (poly + 0.5) * cell + lo


def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def perimeter(poly):
    return float(np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1).sum())


def room_shape(xy):
    """Rectangle unless the occupancy polygon says the room is clearly not one (L-shape, alcove)."""
    rot, rect = manhattan_box(xy)
    poly = room_polygon(xy @ rot)  # in rectangle-aligned coords
    rect_r = rect @ rot
    rect_area = shoelace(rect_r)
    if poly is not None and len(poly) > 4 and shoelace(poly) < 0.85 * rect_area:
        return rot, poly, f"polygon ({len(poly)} vertices) from wall occupancy; room is not rectangular"
    note = "rectangle from the outermost wall planes" + ("" if poly is not None else " (some wall segments unseen)")
    return rot, rect_r, note


# ---------- stage ----------

def run(sweep_dir, reads):
    rec = Recon(sweep_dir)
    samples = marker_scale_samples(sweep_dir, rec)
    out = {"scale": None, "scale_method": "", "scale_samples": samples, "confidence": 0.0, "notes": []}
    if samples:
        s = np.array([x["scale"] for x in samples])
        scale = float(np.median(s))
        spread = float(np.median(np.abs(s - scale)) / scale)
        out["scale_method"] = (f"printed ArUco marker, {settings.MARKER_SIZE_CM} cm side, {len(s)} detections, "
                               f"median abs deviation {spread:.1%}")
        scale_conf = float(np.clip(1 - spread * 5, 0.2, 1.0))
        if rec.metric:  # independent check: a metric model should need a correction near 1.0
            out["model_metric_check"] = {"marker_correction": round(scale, 3), "agrees": abs(scale - 1) <= 0.10}
            out["scale_method"] += f"; MapAnything metric output agrees within {abs(scale - 1):.1%}"
            if abs(scale - 1) > 0.10:
                out["notes"].append(f"marker and metric model disagree on scale by {abs(scale - 1):.0%}")
                scale_conf *= 0.7
    elif rec.metric:
        scale = 1.0
        out["scale_method"] = "MapAnything metric reconstruction (no marker seen; not independently anchored)"
        out["notes"].append("no marker seen; scale from the metric model only")
        scale_conf = 0.5
    else:
        scale, n = book_prior_scale(rec, reads)
        if scale is None:
            out["notes"].append(f"no marker seen and only {n} measurable spines: no metric scale")
            return out
        out["scale_method"] = (f"standard book-format prior: median of {n} upright spines = "
                               f"{settings.BOOK_PRIOR_HEIGHT_CM} cm (no marker seen)")
        out["notes"].append("no marker seen; scale from book-format prior")
        scale_conf = 0.3
    out["scale"] = scale

    pts = np.concatenate([rec.frame_points(i) for i in range(len(rec.frames))]) * scale
    down = np.mean([rec.E[i][:, :3].T @ [0, 1, 0] for i in range(len(rec.frames))], axis=0)
    up, floor_z, ceil_z = floor_ceiling(pts, -down / np.linalg.norm(down))
    out["up"] = up.tolist()
    if floor_z is None:
        out["notes"].append("floor plane not found: sweep did not show enough floor")
        return out
    e1 = np.cross(up, [1, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    h = pts @ up - floor_z
    top = (ceil_z - floor_z - 0.15) if ceil_z is not None else 2.4
    wall = pts[(h > 0.3) & (h < top)]
    xy = np.c_[wall @ e1, wall @ e2]
    rot, poly, shape_note = room_shape(xy)
    # bake the room rotation into the basis so room x runs along the length
    e1, e2 = (np.c_[e1, e2] @ rot).T
    lw = sorted([np.ptp(poly[:, 0]), np.ptp(poly[:, 1])], reverse=True)
    height = (ceil_z - floor_z) if ceil_z is not None else None
    if height is None:
        out["notes"].append("ceiling not seen: height and wall area left empty")
    area, per = shoelace(poly), perimeter(poly)
    if lw[1] < settings.MIN_ROOM_SIDE_M:  # one wall seen: the 'room' is a sliver, not a measurement
        out["notes"].append(f"room extent not captured (narrowest side {lw[1]:.2f} m): dimensions and areas left empty")
        out.update({"e1": e1.tolist(), "e2": e2.tolist(), "floor_z": floor_z, "shape_method": "not captured",
                    "length_m": None, "width_m": None, "height_m": round(height, 2) if height else None,
                    "floor_area_m2": None, "perimeter_m": None, "wall_area_m2": None, "confidence": 0.0})
        return out
    out.update({
        "e1": e1.tolist(), "e2": e2.tolist(), "floor_z": floor_z,
        "polygon_m": poly.round(3).tolist(), "shape_method": shape_note,
        "length_m": round(lw[0], 2), "width_m": round(lw[1], 2),
        "height_m": round(height, 2) if height else None,
        "floor_area_m2": round(area, 2), "perimeter_m": round(per, 2),
        "wall_area_m2": round(per * height, 2) if height else None,
        "confidence": round(scale_conf * (1.0 if height else 0.6) * (0.9 if "rectangle" in shape_note else 0.75), 2),
    })
    return out


class Geo:
    """Room-frame measurements for later stages. ok=False when no metric scale exists."""

    def __init__(self, sweep_dir):
        g = json.loads((sweep_dir / "geometry.json").read_text())
        self.g = g
        self.ok = bool(g.get("scale")) and "e1" in g
        if self.ok:
            self.rec = Recon(sweep_dir)
            self.s, self.z0 = g["scale"], g["floor_z"]
            self.B = np.c_[g["e1"], g["e2"], g["up"]]

    def room(self, P):
        q = np.atleast_2d(P) * self.s @ self.B
        q[:, 2] -= self.z0
        return q

    def spine(self, frame, box_2d, orientation):
        i = self.rec.idx.get(frame)
        if i is None:
            return None
        w, h = self.rec.orig[i]
        m = spine_raw(self.rec, i, box_px(box_2d, w, h), orientation)
        if m is None:
            return None
        height, thick, ctr, bot = m
        return {"height_cm": height * self.s * 100, "thickness_cm": thick * self.s * 100,
                "center": self.room(ctr)[0], "bottom_z": float(self.room(bot)[0, 2])}

    def extent(self, frame, box_2d):
        """Object center and w/h/d in cm from the 3D points inside its box (robust percentiles)."""
        i = self.rec.idx.get(frame)
        if i is None:
            return None
        w, h = self.rec.orig[i]
        P = self.rec.box_points(i, box_px(box_2d, w, h))
        if len(P) < 20:
            return None
        Q = self.room(P)
        # ponytail: box includes some background pixels; p10-p90 trims most of it
        zlo, zhi = np.percentile(Q[:, 2], [10, 90])
        xy = Q[:, :2] - np.median(Q[:, :2], 0)
        axes = np.linalg.svd(xy, full_matrices=False)[2]
        span = [np.subtract(*np.percentile(xy @ a, [90, 10])) for a in axes]
        return {"center": np.median(Q, 0), "w": span[0] * 100, "d": span[1] * 100, "h": (zhi - zlo) * 100}
