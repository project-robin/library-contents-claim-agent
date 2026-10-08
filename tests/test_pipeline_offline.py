"""End-to-end without network: a synthetic 4 x 3 x 2.6 m room rendered as VGGT-style depth (arbitrary scale),
a printed marker and a row of spines on one wall. Runs geometry -> packet with API calls stubbed."""
import json

import cv2
import numpy as np

from pipeline import books, drex, run, settings, spine_ocr, vision
from pipeline.geometry import ARUCO

L, W, H, K_SCALE = 4.0, 3.0, 2.6, 0.37  # K_SCALE: reconstruction units are not metres
FW, FH, GW, GH = 1920, 1080, 518, 294
K = np.array([[400.0, 0, GW / 2], [0, 400.0, GH / 2], [0, 0, 1]])
CAM = np.array([2.0, 1.5, 1.4])
N_BOOKS, BOOK_T, BOOK_H, BOOK_Z0 = 30, 0.03, 0.22, 1.0


def pose(yaw, pitch):
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])  # rows: x right, y down, z forward
    return R, -R @ (CAM * K_SCALE)


def depth_map(R):
    v, u = np.mgrid[0:GH, 0:GW] + 0.5
    dirs = (np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T) @ R  # world dirs, cam z = 1
    c, hi = CAM * K_SCALE, np.array([L, W, H]) * K_SCALE
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs > 0, (hi - c) / dirs, -c / dirs)
    return np.nanmin(np.where(t > 0, t, np.inf), -1)


def project(R, t, P):
    cam = (np.atleast_2d(P) * K_SCALE) @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:] * [FW / GW, FH / GH]


def box2d(R, t, corners):
    p = project(R, t, corners)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return [int(y0 / FH * 1000), int(x0 / FW * 1000), int(y1 / FH * 1000), int(x1 / FW * 1000)]


def make_sweep(sd):
    (sd / "frames").mkdir(parents=True)
    (sd / "reads").mkdir()
    views = [(np.radians(a), 0.0) for a in range(0, 360, 30)] + [(np.radians(a), np.radians(-55)) for a in (0, 90, 180, 270)] \
        + [(np.radians(a), np.radians(60)) for a in (0, 90, 180, 270)] + [(np.radians(a), 0.0) for a in (-6, 6)]
    marker = cv2.aruco.generateImageMarker(ARUCO, 0, 400, borderBits=1)
    m_world = np.array([[L, 2.10, 1.35], [L, 1.95, 1.35], [L, 1.95, 1.20], [L, 2.10, 1.20]])  # 15 cm on the wall x=L
    names, E, D, frames_log, kf = [], [], [], [], []
    for n, (yaw, pitch) in enumerate(views, 1):
        R, t = pose(yaw, pitch)
        name = f"{n:06d}.jpg"
        img = np.full((FH, FW), 255, np.uint8)
        reads = {"spines": [], "items": [], "issue": ""}
        if pitch == 0 and abs(yaw) < np.radians(10):
            src = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
            Hm = cv2.getPerspectiveTransform(src, np.float32(project(R, t, m_world)))
            warped = cv2.warpPerspective(marker, Hm, (FW, FH), borderValue=255)
            img = np.minimum(img, warped)
            for b in range(N_BOOKS):  # spines on wall x=L, y from 1.30 down
                y1, y0 = 1.30 - b * BOOK_T, 1.30 - (b + 1) * BOOK_T
                corners = [[L, y0, BOOK_Z0], [L, y1, BOOK_Z0], [L, y0, BOOK_Z0 + BOOK_H], [L, y1, BOOK_Z0 + BOOK_H]]
                legible = b < 10
                reads["spines"].append({"box_2d": box2d(R, t, corners), "orientation": "upright",
                                        "legibility": "full" if legible else "none",
                                        "title": f"Volume {b}" if legible else "", "author": "Test Author" if legible else "",
                                        "publisher": "", "marks": []})
            lamp = [[L, 0.3, 0.8], [L, 0.6, 0.8], [L, 0.3, 1.4], [L, 0.6, 1.4]]
            reads["items"].append({"box_2d": box2d(R, t, lamp), "category": "lighting", "description": "floor lamp",
                                   "material": "metal", "brand_model": "", "confidence": 0.9})
        cv2.imwrite(str(sd / "frames" / name), img)
        (sd / "reads" / f"{name}.json").write_text(json.dumps(reads))
        names.append(name), E.append(np.c_[R, t]), D.append(depth_map(R))
        frames_log.append({"file": name, "t": float(n), "key": True})
        kf.append({"frame": name, "t": float(n)})
    np.savez_compressed(sd / "recon.npz", frames=np.array(names), orig_wh=np.array([[FW, FH]] * len(names)),
                        extrinsic=np.array(E), intrinsic=np.array([K] * len(names)),
                        depth=np.array(D, np.float16), conf=np.ones((len(names), GH, GW), np.float16))
    (sd / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in frames_log))
    (sd / "keyframes.json").write_text(json.dumps(kf))
    (sd / "sweep.json").write_text(json.dumps({"id": "synthetic", "captured_at": "2026-10-06T00:00:00+00:00",
                                               "device": "test", "country": "IN", "currency": "INR",
                                               "duration_s": 60}))


def test_synthetic_room_end_to_end(tmp_path, monkeypatch):
    sd = tmp_path / "sweep"
    make_sweep(sd)
    monkeypatch.setattr(vision, "read_crops", lambda js: [{"title": "", "author": "", "publisher": "",
                                                           "legibility": "none"} for _ in js])
    monkeypatch.setattr(vision, "check_identity", lambda c, t, a: {"verdict": "match", "reason": "stub"})
    monkeypatch.setattr(books, "_gbooks", lambda t, a, free=False: [{"title": t, "subtitle": "", "authors": [a], "publisher": "",
                                                         "year": "2001", "isbn": "", "url": "u", "source": "stub"}])
    monkeypatch.setattr(books, "_openlibrary", lambda t, a: [])
    monkeypatch.setattr(drex, "ask", lambda state, qs: {k: 0.95 for k in qs})
    monkeypatch.setattr(spine_ocr, "read", lambda crops: [{"words": [], "rotation": None} for _ in crops])
    monkeypatch.setattr(settings, "require", lambda what: None)
    monkeypatch.setattr(settings, "SERPER_API_KEY", "")  # no network: pricing finds nothing and must invent nothing
    p = run.run(sd, start="geometry", log=lambda s: None)

    r = p["room"]
    assert "marker" in r["scale_method"]
    assert abs(r["floor_area_m2"] - L * W) / (L * W) < 0.05, r
    assert abs(r["height_m"] - H) < 0.08, r
    assert abs(r["wall_area_m2"] - 2 * (L + W) * H) / (2 * (L + W) * H) < 0.08, r
    t = p["totals"]
    assert t["book_count"] == N_BOOKS, t
    assert t["books_identified"] == 10 and t["books_unidentified"] == N_BOOKS - 10
    assert abs(t["shelf_run_m"] - N_BOOKS * BOOK_T) < 0.1
    hs = [b["spine_height_cm"] for b in p["books"]]
    ts = [b["spine_thickness_cm"] for b in p["books"]]
    assert abs(np.median(hs) - 22) < 2 and abs(np.median(ts) - 3) < 0.5, (hs[:5], ts[:5])
    assert all(b["replacement_cost"]["amount"] is None for b in p["books"])  # no keys -> no prices, never invented
    assert t["books_replacement_cost"] is None and t["excluded_from_totals"] >= N_BOOKS
    assert len(p["items"]) == 1 and p["items"][0]["category"] == "lighting"
    assert any("unidentified" in q["reason"] or "not readable" in q["reason"] for q in p["review_queue"])
    for b in p["books"]:
        assert (sd / b["frame_ref"].split("#")[0]).exists() and (sd / b["crop"]).exists()
    assert (sd / "report.html").read_text().count("<tr>") > N_BOOKS
