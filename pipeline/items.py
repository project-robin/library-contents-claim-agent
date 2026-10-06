"""Non-book contents: per-frame item detections -> one line per physical object with 3D dimensions."""
from collections import Counter

import numpy as np

from pipeline.books import crop, frame_times, load_notes, nearest_det


def run(sweep_dir, reads, geo):
    times = frame_times(sweep_dir)
    dets = []
    for f, r in reads.items():
        for k, it in enumerate(r["items"]):
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
    for note in load_notes(sweep_dir):
        if note["kind"] == "fact":  # set_item_fact from the live agent, anchored to the live detection id
            c = det_to_c.get(note.get("det_id"))
        elif note["kind"] == "note" and note.get("target") == "item":
            c = det_to_c.get(nearest_det(reads, times, note["t"], "items"))
        else:
            continue
        if c:
            c.setdefault("facts", []).append(note)

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
    shelving = [i for i in items if i["category"] == "shelving"]
    shelved_wall = sum(i["dimensions_cm"]["w"] * i["dimensions_cm"]["h"] for i in shelving) / 1e4
    return {"items": items, "stats": {"detections": len(dets),
                                      "detections_without_3d": sum("center" not in d for d in dets),
                                      "shelved_wall_area_m2": round(shelved_wall, 2) if shelving else None}}
