"""Post-sweep pipeline. Each stage reads the previous stage's JSON and writes its own, so any stage can be rerun:

    uv run python -m pipeline.run sweeps/<id> [--from books]
"""
import argparse
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from pipeline import books, frames, geometry, items, metrics, packet, pricing, review, settings, vision

STAGES = ["keyframes", "reads", "recon", "geometry", "books", "checker", "items", "pricing", "packet"]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                               default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def load(path):
    return json.loads(path.read_text())


def keyframes(sd, recompute):
    rows = [json.loads(x) for x in (sd / "frames.jsonl").read_text().splitlines() if x]
    if recompute or not any(r.get("key") for r in rows):
        return frames.select_keyframes(sd)
    return [{"frame": r["file"], "t": r["t"]} for r in rows if r.get("key")]


def read_missing(sd, kfs):
    (sd / "reads").mkdir(exist_ok=True)
    todo = [k["frame"] for k in kfs if not (sd / "reads" / f"{k['frame']}.json").exists()]

    def one(f):
        dump(sd / "reads" / f"{f}.json", vision.read_frame((sd / "frames" / f).read_bytes()))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    return {k["frame"]: load(sd / "reads" / f"{k['frame']}.json") for k in kfs}


def recon(sd, kfs):
    from gpu.modal_recon import remote  # imported lazily: only this stage needs Modal

    imgs = {k["frame"]: cv2.imread(str(sd / "frames" / k["frame"])) for k in kfs}
    size = Counter((im.shape[1], im.shape[0]) for im in imgs.values()).most_common(1)[0][0]
    # ponytail: frames from a different phone orientation are left out of 3D (still read for text)
    names = [f for f, im in imgs.items() if (im.shape[1], im.shape[0]) == size]
    if len(names) > settings.MAX_RECON_FRAMES:
        names = [names[int(i)] for i in np.linspace(0, len(names) - 1, settings.MAX_RECON_FRAMES)]
    gw, gh = geometry.recon_size(*size)
    jpegs = [cv2.imencode(".jpg", cv2.resize(imgs[f], (gw, gh), interpolation=cv2.INTER_AREA),
                          [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for f in names]
    t0 = time.perf_counter()
    z = np.load(io.BytesIO(remote().run.remote(jpegs)))
    metrics.count("gpu_seconds", time.perf_counter() - t0)
    np.savez_compressed(sd / "recon.npz", frames=np.array(names), orig_wh=np.array([size] * len(names)),
                        **{k: z[k] for k in ("extrinsic", "intrinsic", "depth", "conf")})


def run(sd, start="keyframes", recompute_keyframes=False, log=print):
    sd = Path(sd)
    sweep = load(sd / "sweep.json")
    country = sweep.get("country") or settings.DEFAULT_COUNTRY
    go = lambda s: STAGES.index(s) >= STAGES.index(start)

    if go("keyframes"):
        with metrics.stage("keyframes"):
            dump(sd / "keyframes.json", keyframes(sd, recompute_keyframes))
    kfs = load(sd / "keyframes.json")
    log(f"{len(kfs)} keyframes")

    def timed(name, fn):
        with metrics.stage(name):
            return fn()

    with ThreadPoolExecutor(2) as ex:  # GPU reconstruction and spine reading run in parallel
        f_reads = ex.submit(timed, "reads", lambda: read_missing(sd, kfs)) if go("reads") else None
        f_recon = ex.submit(timed, "recon", lambda: recon(sd, kfs)) if go("recon") else None
        if f_recon:
            f_recon.result()
        reads = f_reads.result() if f_reads else {
            k["frame"]: load(sd / "reads" / f"{k['frame']}.json") for k in kfs
            if (sd / "reads" / f"{k['frame']}.json").exists()}
    log("reads + recon done")

    if go("geometry"):
        with metrics.stage("geometry"):
            dump(sd / "geometry.json", geometry.run(sd, reads))
    geo_json = load(sd / "geometry.json")
    log(f"geometry: scale={geo_json.get('scale')} floor={geo_json.get('floor_area_m2')} m2")
    geo = geometry.Geo(sd)

    if go("books"):
        with metrics.stage("books"):
            dump(sd / "books.json", books.run(sd, reads, geo))
    books_out = load(sd / "books.json")
    if go("checker"):
        with metrics.stage("checker"):
            review.checker(sd, books_out["books"])
            dump(sd / "books.json", books_out)
    log(f"books: {len(books_out['books'])}")

    if go("items"):
        with metrics.stage("items"):
            dump(sd / "items.json", items.run(sd, reads, geo))
    items_out = load(sd / "items.json")

    if go("pricing"):
        with metrics.stage("pricing"):
            dump(sd / "prices.json", pricing.run(sd, books_out["books"], items_out["items"], country))
    prices = load(sd / "prices.json")
    log("pricing done")

    with metrics.stage("packet"):
        q = review.queue(geo_json, books_out["books"], items_out["items"], prices["books"], prices["items"],
                         books_out["stats"])
        extra = {"keyframes": len(kfs)}
        if sweep.get("t_end_epoch"):
            extra["time_to_packet_s"] = round(time.time() - sweep["t_end_epoch"], 1)
        m = metrics.save(sd / "metrics.json", extra)
        p = packet.build(sweep, geo_json, books_out, items_out, prices, q, m)
        dump(sd / "claim_packet.json", p)
        svg = packet.floorplan_svg(geo_json, books_out, items_out)
        (sd / "floorplan.svg").write_text(svg)
        (sd / "report.html").write_text(packet.report_html(p, svg))
    metrics.save(sd / "metrics.json", extra)
    log(f"packet: {sd / 'claim_packet.json'}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--from", dest="start", default="keyframes", choices=STAGES)
    ap.add_argument("--recompute-keyframes", action="store_true")
    a = ap.parse_args()
    run(a.sweep_dir, a.start, a.recompute_keyframes)
