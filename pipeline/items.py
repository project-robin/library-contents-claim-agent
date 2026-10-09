"""Non-book contents: per-frame item detections -> one line per physical object with 3D dimensions."""
from collections import Counter

import cv2
import numpy as np

from pipeline import geometry, reconcile
from pipeline.books import crop, frame_times, load_notes, nearest_det

MARKER_SHARE = 0.08  # a box whose area is at least 8% marker is the marker card (or its clipboard), not an item


def shows_marker(sweep_dir, frame, box_2d, cache, det):
    """The live sweep listed the printed scale marker twice as art ('framed artwork depicting a black and white
    geometric pattern'). OpenCV finds the ArUco marker inside the box; a shelf with the card leaning on it is far
    larger than the card, so it stays."""
    if frame not in cache:
        cache[frame] = cv2.imread(str(sweep_dir / "frames" / frame), cv2.IMREAD_GRAYSCALE)
    img = cache[frame]
    if img is None:
        return False
    h, w = img.shape
    x0, y0, x1, y1 = (int(v) for v in geometry.box_px(box_2d, w, h))
    sub = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if sub.size == 0:
        return False
    corners, ids = geometry.detect_marker(det, sub)
    if ids is None:
        return False
    marker = max(cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in corners)
    return marker >= MARKER_SHARE * sub.shape[0] * sub.shape[1]


def run(sweep_dir, reads, geo):
    times = frame_times(sweep_dir)
    dets, marker_dets, gray = [], 0, {}
    aruco = cv2.aruco.ArucoDetector(geometry.ARUCO, cv2.aruco.DetectorParameters())
    for f, r in reads.items():
        for k, it in enumerate(r["items"]):
            if shows_marker(sweep_dir, f, it["box_2d"], gray, aruco):
                marker_dets += 1
                continue
            d = {"id": f"{f}:{k}", "frame": f, "t": times.get(f), **it}
            e = geo.extent(f, it["box_2d"]) if geo.ok else None
            if e:
                d.update(e)
            dets.append(d)

    clusters = []  # ponytail: greedy, same category within 0.6 m = same object
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        if "center" not in d:
            continue
        c = next((c for c in clusters if c["category"] == d["category"]
                  and np.linalg.norm(c["center"] - d["center"]) < 0.6), None)
        if c:
            c["members"].append(d)
        else:
            clusters.append({"category": d["category"], "center": d["center"], "members": [d]})

    det_to_c = {m["id"]: c for c in clusters for m in c["members"]}
    unmatched = []
    for note in load_notes(sweep_dir):
        if note["kind"] in ("fact", "removed"):  # set_item_fact / remove_item, anchored to the live detection id
            c = det_to_c.get(note.get("det_id"))
            if not c and note["kind"] == "removed":
                unmatched.append(note)
        elif note["kind"] == "note" and note.get("target") == "item":
            c = det_to_c.get(nearest_det(reads, times, note["t"], "items"))
        else:
            continue
        if c:
            c.setdefault("facts", []).append(note)

    removed = []
    for c in clusters:
        if gone := [f for f in c.get("facts", []) if f["kind"] == "removed"]:
            removed.append({"description": c["members"][0]["description"], "reason": gone[-1].get("text", "")})
    removed += [{"description": n.get("description", n.get("item_id", "")), "reason": n.get("text", ""),
                 "no_line": True} for n in unmatched]
    clusters = [c for c in clusters if not any(f["kind"] == "removed" for f in c.get("facts", []))]

    (sweep_dir / "crops").mkdir(exist_ok=True)
    items = []
    for n, c in enumerate(clusters, 1):
        iid = f"i_{n:03d}"
        ms = c["members"]
        best = max(ms, key=lambda m: m["confidence"])
        crop(sweep_dir, best["frame"], best["box_2d"], sweep_dir / "crops" / f"{iid}.jpg", pad=0.03)
        med = lambda k: round(float(np.median([m[k] for m in ms])), 0)
        pick = lambda k: (Counter(m[k] for m in ms if m[k]).most_common(1) or [("", 0)])[0][0]
        facts = c.get("facts", [])
        is_print = next((f["is_print"] for f in reversed(facts) if f.get("is_print") is not None), None)
        items.append({
            "id": iid, "category": c["category"], "description": pick("description"), "material": pick("material"),
            "brand_model": pick("brand_model"), "frame_ref": f"frames/{best['frame']}#t={best['t']}",
            "crop": f"crops/{iid}.jpg", "views": len(ms),
            "dimensions_cm": {"w": med("w"), "h": med("h"), "d": med("d")},
            "confidence": round(float(np.mean([m["confidence"] for m in ms])) * min(1.0, 0.6 + 0.2 * len(ms)), 2),
            "is_print": is_print, "notes": [f.get("text", "") for f in facts if f.get("text")],
            "_center": [round(float(v), 3) for v in c["center"]],
        })
    # wide views for the check: the read frames that saw the most items
    wide = sorted(reads, key=lambda f: -len(reads[f]["items"]))[:2]
    items, rec = reconcile.run_items(sweep_dir, items, wide)
    shelving = [i for i in items if i["category"] == "shelving"]
    shelved_wall = sum(i["dimensions_cm"]["w"] * i["dimensions_cm"]["h"] for i in shelving) / 1e4
    return {"items": items, "stats": {"detections": len(dets),
                                      "detections_without_3d": sum("center" not in d for d in dets),
                                      "shelved_wall_area_m2": round(shelved_wall, 2) if shelving else None,
                                      "marker_detections_dropped": marker_dets, "removed_by_user": removed,
                                      "reconciled": [{"id": i, "action": k, "into": into, "reason": r}
                                                     for i, k, into, r in rec]}}
