"""Post-sweep pipeline. Each stage reads the previous stage's JSON and writes its own, so any stage can be rerun:

    uv run python -m pipeline.run sweeps/<id> [--from books]
    uv run python -m pipeline.run sweeps/photo_B --from spines --skip recon,geometry --until checker  # a still photo
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

from pipeline import books, frames, geometry, items, metrics, packet, pricing, review, sam, settings, vision
from pipeline.contract import ClaimPacket

STAGES = ["keyframes", "reads", "spines", "recon", "geometry", "books", "checker", "items", "pricing", "packet"]


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
    return {k["frame"]: vision.clean_read(load(sd / "reads" / f"{k['frame']}.json")) for k in kfs}


def load_spines(sd, kfs):
    """SAM spines per keyframe from spines/<frame>.json, or None = use the vision model's spines (SAM off, or more
    than 10% of keyframes without a SAM answer: then the boxes would come from two sources)."""
    if settings.SPINE_SOURCE != "sam":
        return None
    have = {k["frame"]: sd / "spines" / f"{k['frame']}.json" for k in kfs}
    have = {f: p for f, p in have.items() if p.exists()}
    if len(kfs) - len(have) > 0.1 * len(kfs):
        return None
    return {f: load(p)["spines"] for f, p in have.items()}


def segment_spines(sd, kfs, log=print):
    """SAM 3.1 boxes for every keyframe not yet cached. A failed frame is not cached, so a rerun retries it."""
    if not sam.available():
        log("spines: no META_API_KEY or SPINE_SOURCE=gemini, using the vision model's spines")
        return None
    (sd / "spines").mkdir(exist_ok=True)
    todo = [k["frame"] for k in kfs if not (sd / "spines" / f"{k['frame']}.json").exists()]

    def one(f):
        try:
            boxes = sam.segment((sd / "frames" / f).read_bytes())
        except Exception as e:  # any API failure: this frame has no SAM spines
            log(f"spines: SAM failed on {f}: {e}")
            return False
        dump(sd / "spines" / f"{f}.json", {"source": "sam", "spines": sam.to_spines(boxes)})
        return True

    with ThreadPoolExecutor(settings.SAM_CONCURRENCY) as ex:
        failed = sum(not ok for ok in ex.map(one, todo))
    if failed:
        metrics.count("sam_failures", failed)
    out = load_spines(sd, kfs)
    if out is None:
        log(f"spines: SAM failed on {failed}/{len(kfs)} keyframes, using the vision model's spines")
    return out


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
    z = np.load(io.BytesIO(remote(settings.RECON_MODEL).run.remote(jpegs)))
    metrics.count("gpu_seconds", time.perf_counter() - t0)
    np.savez_compressed(sd / "recon.npz", frames=np.array(names), orig_wh=np.array([size] * len(names)),
                        model=settings.RECON_MODEL, **{k: z[k] for k in ("extrinsic", "intrinsic", "depth", "conf", "metric")})


def run(sd, start="keyframes", recompute_keyframes=False, log=print, skip=(), until=None):
    """skip: stages not run (their saved outputs are loaded). until: stop after this stage, before the packet."""
    last = STAGES.index(until or "packet")
    settings.require("pipeline" if last >= STAGES.index("pricing") else "books")
    sd = Path(sd)
    sweep = load(sd / "sweep.json")
    country = sweep.get("country") or settings.DEFAULT_COUNTRY
    go = lambda s: STAGES.index(s) >= STAGES.index(start) and s not in skip
    past = lambda s: STAGES.index(s) > last  # stage s comes after `until`

    def stop(ret):
        metrics.save(sd / "metrics.json")
        log(f"stopped after {until}")
        return ret

    if go("keyframes"):
        with metrics.stage("keyframes"):
            dump(sd / "keyframes.json", keyframes(sd, recompute_keyframes))
    kfs = load(sd / "keyframes.json")
    log(f"{len(kfs)} keyframes")

    def timed(name, fn):
        with metrics.stage(name):
            return fn()

    with ThreadPoolExecutor(3) as ex:  # GPU reconstruction, spine reading and SAM segmentation run in parallel
        f_reads = ex.submit(timed, "reads", lambda: read_missing(sd, kfs)) if go("reads") else None
        f_spines = ex.submit(timed, "spines", lambda: segment_spines(sd, kfs, log)) if go("spines") else None
        f_recon = ex.submit(timed, "recon", lambda: recon(sd, kfs)) if go("recon") else None
        if f_recon:
            f_recon.result()
        reads = f_reads.result() if f_reads else {
            k["frame"]: vision.clean_read(load(sd / "reads" / f"{k['frame']}.json")) for k in kfs
            if (sd / "reads" / f"{k['frame']}.json").exists()}
        spines = f_spines.result() if f_spines else load_spines(sd, kfs)
    log(f"reads + recon done; spines from {'SAM' if spines is not None else 'the vision model'}")
    if past("geometry"):
        return stop(None)

    if go("geometry"):
        with metrics.stage("geometry"):
            dump(sd / "geometry.json", geometry.run(sd, reads))
    geo_json = load(sd / "geometry.json")
    log(f"geometry: scale={geo_json.get('scale')} floor={geo_json.get('floor_area_m2')} m2")
    geo = geometry.Geo(sd)
    if past("books"):
        return stop(None)

    if go("books"):
        with metrics.stage("books"):
            dump(sd / "books.json", books.run(sd, reads, geo, spines))
    books_out = load(sd / "books.json")
    if past("checker"):
        return stop(books_out)
    if go("checker"):
        with metrics.stage("checker"):
            review.checker(sd, books_out["books"])
            dump(sd / "books.json", books_out)
    log(f"books: {len(books_out['books'])}")
    if past("items"):
        return stop(books_out)

    if go("items"):
        with metrics.stage("items"):
            dump(sd / "items.json", items.run(sd, reads, geo))
    items_out = load(sd / "items.json")
    if past("pricing"):
        return stop(books_out)

    if go("pricing"):
        with metrics.stage("pricing"):
            dump(sd / "prices.json", pricing.run(sd, books_out["books"], items_out["items"], country))
    prices = load(sd / "prices.json")
    log("pricing done")
    if past("packet"):
        return stop(books_out)

    with metrics.stage("packet"):
        q = review.queue(geo_json, books_out["books"], items_out["items"], prices["books"], prices["items"],
                         books_out["stats"])
        extra = {"keyframes": len(kfs)}
        if sweep.get("t_end_epoch"):
            extra["time_to_packet_s"] = round(time.time() - sweep["t_end_epoch"], 1)
        m = metrics.save(sd / "metrics.json", extra)
        p = packet.build(sweep, geo_json, books_out, items_out, prices, q, m)
        ClaimPacket.model_validate(p)  # contract check: a dropped field or unsourced price fails here, not in review
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
    ap.add_argument("--skip", default="", help="comma list of stages not to run, e.g. recon,geometry for a still")
    ap.add_argument("--until", choices=STAGES, help="stop after this stage")
    a = ap.parse_args()
    skip = [s for s in a.skip.split(",") if s]
    if bad := set(skip) - set(STAGES):
        ap.error(f"unknown stage(s) in --skip: {', '.join(sorted(bad))}")
    run(a.sweep_dir, a.start, a.recompute_keyframes, skip=skip, until=a.until)
