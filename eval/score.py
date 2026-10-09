"""Score a claim packet against hand-collected ground truth, one row per pass bar.

    uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/ [--md]

Ground truth CSVs (see ground_truth/*.csv headers). Nothing here is tuned per room.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from pipeline.books import related, sim

TITLE_MATCH = 0.85


def tsim(a, b):
    """1.0 when one title starts or ends the other ('Othello' / 'Othello - William Shakespeare', 'Raavan' / 'Raavan:
    Enemy of Aryavarta'): the packet appends an author or drops a subtitle. Otherwise string similarity. A title
    inside a longer one ('Ikigai' in 'Japonisme: Ikigai, ...') still does not match."""
    return 1.0 if related(a, b) else sim(a, b)


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def match_titles(gt_titles, packet_books):
    """Greedy one-to-one match on title similarity. Returns {gt_index: book}."""
    pairs = sorted(((tsim(g, b["title"]), gi, bi) for gi, g in enumerate(gt_titles)
                    for bi, b in enumerate(packet_books) if b["title"]), reverse=True)
    used_g, used_b, out = set(), set(), {}
    for s, gi, bi in pairs:
        if s < TITLE_MATCH:
            break
        if gi not in used_g and bi not in used_b:
            used_g.add(gi), used_b.add(bi)
            out[gi] = packet_books[bi]
    return out


def pct(a, b):
    return abs(a - b) / b * 100 if b else float("nan")


def score(packet, gt):
    res = []
    add = lambda name, bar, value, ok, detail="": res.append(
        {"measure": name, "pass_bar": bar, "value": value, "pass": ok, "detail": detail})
    books = packet["books"]
    gt_books = rows(gt / "books.csv")

    # count
    n_true = len(gt_books)
    if n_true:
        e = pct(packet["totals"]["book_count"], n_true)
        add("Book count", "within 5%", f"{packet['totals']['book_count']} vs {n_true} ({e:.1f}%)", e <= 5)

    # titles: recall on human-legible spines, confident-wrong over all confident claims
    legible = [b for b in gt_books if b.get("legible", "").strip().lower() in ("y", "yes", "1", "true")]
    if legible:
        m = match_titles([b["title"] for b in legible], books)
        recall = len(m) / len(legible) * 100
        all_gt = [b["title"] for b in gt_books if b["title"]]
        confident = [b for b in books if b["title"] and b["id_confidence"] >= 0.75]
        wrong = [b for b in confident if max((tsim(b["title"], g) for g in all_gt), default=0) < TITLE_MATCH]
        wrong_pct = len(wrong) / max(len(confident), 1) * 100
        add("Title identification", ">=70% correct, <=3% confidently wrong",
            f"{recall:.1f}% correct ({len(m)}/{len(legible)}), {wrong_pct:.1f}% confidently wrong ({len(wrong)}/{len(confident)})",
            recall >= 70 and wrong_pct <= 3, "; ".join(f"{b['id']} '{b['title']}'" for b in wrong[:10]))

    # spine dims on the hand-measured sample
    sp = rows(gt / "spines.csv")
    if sp:
        m = match_titles([s["title"] for s in sp], books)
        errs = []
        for gi, b in m.items():
            if b["spine_height_cm"] and b["spine_thickness_cm"]:
                errs += [pct(b["spine_height_cm"], float(sp[gi]["height_cm"])),
                         pct(b["spine_thickness_cm"], float(sp[gi]["thickness_cm"]))]
        within = sum(e <= 15 for e in errs) / max(len(errs), 1) * 100
        mean = statistics.mean(errs) if errs else float("nan")
        add("Spine dimensions", "within 15% (mean abs error)",
            f"mean {mean:.1f}%, {within:.0f}% of measures within 15%, {len(m)}/{len(sp)} spines matched",
            bool(errs) and mean <= 15)

    # prices
    pr = rows(gt / "prices.csv")
    if pr:
        m = match_titles([p["title"] for p in pr], books)
        errs = [pct(b["replacement_cost"]["amount"], float(pr[gi]["amount"]))
                for gi, b in m.items() if b["replacement_cost"]["amount"]]
        within = sum(e <= 25 for e in errs) / max(len(errs), 1) * 100
        add("Book prices", "within 25% of hand-checked (same source type)",
            f"median error {statistics.median(errs):.1f}%, {within:.0f}% within 25%, {len(errs)}/{len(pr)} priced"
            if errs else f"0/{len(pr)} priced", bool(errs) and statistics.median(errs) <= 25)

    # items: category recall (multiset)
    it = rows(gt / "items.csv")
    if it:
        pool = [i["category"] for i in packet["items"]]
        found = 0
        for g in it:
            if g["category"] in pool:
                pool.remove(g["category"])
                found += 1
        r = found / len(it) * 100
        add("Non-book items", ">=80% found with correct category", f"{r:.0f}% ({found}/{len(it)})", r >= 80)

    room = {r["measure"]: float(r["value"]) for r in rows(gt / "room.csv") if r.get("value")}
    for key, name, bar in (("floor_area_m2", "Floor area", 10), ("wall_area_m2", "Wall area", 15)):
        if key in room:
            v = packet["room"].get(key)
            e = pct(v, room[key]) if v else float("nan")
            add(name, f"within {bar}%", f"{v} vs {room[key]} m² ({e:.1f}%)" if v else "not measured",
                bool(v) and e <= bar)
    for key in ("length_m", "width_m", "height_m"):
        if key in room and packet["room"].get(key):
            add(f"Room {key}", "(diagnostic)", f"{packet['room'][key]} vs {room[key]} ({pct(packet['room'][key], room[key]):.1f}%)", None)

    ttp = packet.get("metrics", {}).get("time_to_packet_s")
    if ttp is not None:
        add("Time to packet", "under 5 minutes", f"{ttp:.0f} s", ttp < 300)
    return res


def markdown(res):
    out = ["| Measure | Pass bar | Result | Pass |", "|---|---|---|---|"]
    for r in res:
        out.append(f"| {r['measure']} | {r['pass_bar']} | {r['value']} | "
                   f"{'—' if r['pass'] is None else ('✅' if r['pass'] else '❌')} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("packet")
    ap.add_argument("ground_truth")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    res = score(json.loads(Path(a.packet).read_text()), Path(a.ground_truth))
    print(markdown(res) if a.md else json.dumps(res, indent=1, ensure_ascii=False))
