"""Claim packet: JSON per the output contract, totals computed here in Decimal, plus the HTML report."""
import html
import json
from decimal import Decimal

M2_TO_FT2 = Decimal("10.7639")


def D(x):
    return Decimal(str(x))


def _price(p, kind):
    if kind == "replacement":
        base = {"amount": None, "source": "", "url": "", "retrieved_at": "", "converted": False}
    else:
        base = {"amount": None, "source": "", "url": "", "retrieved_at": "", "condition_assumed": ""}
    return base | (p or {})


def book_line(b, p):
    status = p.get("status", "unidentified")
    return {
        "id": b["id"], "shelf": b["shelf"], "position": b["position"], "frame_ref": b["frame_ref"],
        "status": status, "title": b["title"], "author": b["author"], "edition": b["edition"], "isbn": b["isbn"],
        "spine_height_cm": b["spine_height_cm"], "spine_thickness_cm": b["spine_thickness_cm"],
        "id_confidence": b["id_confidence"],
        "replacement_cost": _price(p.get("replacement_cost"), "replacement"),
        "used_value": _price(p.get("used_value"), "used"),
        # extra evidence fields
        "orientation": b["orientation"], "crop": b["crop"], "views": b["views"], "spine_text": b["spine_text"],
        "catalog": b["catalog"], "checker": b.get("checker"), "marks": b["marks"], "notes": b["notes"],
    }


def item_line(it, p):
    rc = {"low": None, "high": None, "source": "", "url": "", "retrieved_at": ""} | (p.get("replacement_cost") or {})
    return {
        "id": it["id"], "category": it["category"], "description": it["description"],
        "brand_model": it["brand_model"], "frame_ref": it["frame_ref"], "dimensions_cm": it["dimensions_cm"],
        "status": p.get("status", "range"), "replacement_cost": rc, "confidence": it["confidence"],
        "material": it["material"], "crop": it["crop"], "views": it["views"], "is_print": it["is_print"],
        "notes": it["notes"],
    }


def totals(books, items, shelf_run_m):
    """The only place money is added up. Lines without a sourced amount contribute nothing and are counted."""
    rep = sum((D(b["replacement_cost"]["amount"]) for b in books
               if b["status"] == "identified" and b["replacement_cost"]["amount"] is not None), Decimal(0))
    used = sum((D(b["used_value"]["amount"]) for b in books
                if b["status"] == "identified" and b["used_value"]["amount"] is not None), Decimal(0))
    priced_items = [i for i in items if i["status"] in ("priced", "range") and i["replacement_cost"]["low"] is not None]
    lo = sum((D(i["replacement_cost"]["low"]) for i in priced_items), Decimal(0))
    hi = sum((D(i["replacement_cost"]["high"]) for i in priced_items), Decimal(0))
    in_rep = sum(1 for b in books if b["status"] == "identified" and b["replacement_cost"]["amount"] is not None)
    identified = sum(1 for b in books if b["title"])
    return {
        "book_count": len(books), "books_identified": identified, "books_unidentified": len(books) - identified,
        "shelf_run_m": shelf_run_m,
        "books_replacement_cost": float(rep.quantize(Decimal("0.01"))),
        "books_used_value": float(used.quantize(Decimal("0.01"))),
        "items_replacement_cost_low": float(lo.quantize(Decimal("0.01"))),
        "items_replacement_cost_high": float(hi.quantize(Decimal("0.01"))),
        "excluded_from_totals": (len(books) - in_rep) + (len(items) - len(priced_items)),
        "books_needs_appraisal": sum(1 for b in books if b["status"] == "needs_appraisal"),
        "items_needs_appraisal": sum(1 for i in items if i["status"] == "needs_appraisal"),
    }


def build(sweep, geometry, books_out, items_out, prices, review_queue, metrics):
    books = [book_line(b, prices["books"].get(b["id"], {})) for b in books_out["books"] if not b["excluded"]]
    items = [item_line(i, prices["items"].get(i["id"], {})) for i in items_out["items"]]
    g = geometry
    area = g.get("floor_area_m2")
    wall = g.get("wall_area_m2")
    return {
        "sweep": {k: sweep.get(k, "") for k in ("id", "captured_at", "device", "duration_s", "country", "currency")},
        "room": {
            "length_m": g.get("length_m"), "width_m": g.get("width_m"), "height_m": g.get("height_m"),
            "floor_area_m2": area, "wall_area_m2": wall,
            "shelved_wall_area_m2": items_out["stats"]["shelved_wall_area_m2"],
            "scale_method": g.get("scale_method", ""), "confidence": g.get("confidence", 0),
            "floor_area_ft2": float((D(area) * M2_TO_FT2).quantize(Decimal("0.1"))) if area else None,
            "wall_area_ft2": float((D(wall) * M2_TO_FT2).quantize(Decimal("0.1"))) if wall else None,
            "shelved_wall_area_ft2": float((D(items_out["stats"]["shelved_wall_area_m2"]) * M2_TO_FT2)
                                           .quantize(Decimal("0.1"))) if items_out["stats"]["shelved_wall_area_m2"] else None,
            "shape_method": g.get("shape_method", ""), "perimeter_m": g.get("perimeter_m"),
            "wall_area_note": "gross: perimeter x height; doors and windows not subtracted",
            "scale_samples": len(g.get("scale_samples", [])),
        },
        "books": books,
        "items": items,
        "totals": totals(books, items, books_out["stats"]["shelf_run_m"]),
        "review_queue": review_queue,
        "excluded_by_user": [{"id": b["id"], "shelf": b["shelf"], "reason": b["excluded"]}
                             for b in books_out["books"] if b["excluded"]],
        "locale_comparison": prices["locale_comparison"],
        "metrics": metrics,
    }


# ---------- report ----------

e = lambda x: html.escape("" if x is None else str(x))


def _money(x, cur):
    return "" if x is None else f"{cur} {x:,.2f}"


def _link(text, url):
    return f'<a href="{e(url)}">{e(text)}</a>' if url else e(text)


def floorplan_svg(geometry, books_out, items_out):
    poly = geometry.get("polygon_m")
    if not poly:
        return ""
    px = 90
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y0 = min(xs) - 0.3, min(ys) - 0.3
    W, H = (max(xs) - x0 + 0.3) * px, (max(ys) - y0 + 0.3) * px
    tx = lambda x: (x - x0) * px
    ty = lambda y: H - (y - y0) * px
    pts = " ".join(f"{tx(x):.0f},{ty(y):.0f}" for x, y in poly)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">',
           f'<polygon points="{pts}" fill="#f6f4ef" stroke="#333" stroke-width="3"/>']
    units = {}
    for b in books_out["books"]:
        units.setdefault(b["shelf"].split("-")[0], []).append(b["_center"])
    for u, cs in units.items():
        cx, cy = sum(c[0] for c in cs) / len(cs), sum(c[1] for c in cs) / len(cs)
        out.append(f'<rect x="{tx(cx) - 14:.0f}" y="{ty(cy) - 14:.0f}" width="28" height="28" fill="#8a5a2b"/>'
                   f'<text x="{tx(cx) + 18:.0f}" y="{ty(cy) + 5:.0f}" font-size="13">Unit {e(u)} ({len(cs)} books)</text>')
    for it in items_out["items"]:
        x, y = it["_center"][:2]
        out.append(f'<circle cx="{tx(x):.0f}" cy="{ty(y):.0f}" r="6" fill="#2b6a8a"/>'
                   f'<text x="{tx(x) + 9:.0f}" y="{ty(y) + 4:.0f}" font-size="11">{e(it["id"])} {e(it["category"])}</text>')
    out.append(f'<text x="8" y="{H - 8:.0f}" font-size="11">1 m = {px} px. Positions from the 3D reconstruction.</text></svg>')
    return "".join(out)


def report_html(p, svg):
    cur = p["sweep"]["currency"]
    r, t = p["room"], p["totals"]
    rows_b = "".join(
        f"<tr><td>{e(b['id'])}</td><td>{e(b['shelf'])}/{b['position']}</td>"
        f"<td><a href='{e(b['crop'])}'><img src='{e(b['crop'])}' height='60'></a></td>"
        f"<td>{e(b['status'])}</td><td>{e(b['title'])}<br><small>{e(b['author'])}</small></td>"
        f"<td>{e(b['spine_height_cm'])} × {e(b['spine_thickness_cm'])}</td><td>{b['id_confidence']:.2f}</td>"
        f"<td>{_link(_money(b['replacement_cost']['amount'], cur), b['replacement_cost']['url'])}"
        f"{' (converted)' if b['replacement_cost']['converted'] else ''}<br><small>{e(b['replacement_cost']['source'])}</small></td>"
        f"<td>{_link(_money(b['used_value']['amount'], cur), b['used_value']['url'])}<br>"
        f"<small>{e(b['used_value']['condition_assumed'])}</small></td>"
        f"<td><a href='{e(b['frame_ref'].split('#')[0])}'>{e(b['frame_ref'])}</a></td></tr>" for b in p["books"])
    rows_i = "".join(
        f"<tr><td>{e(i['id'])}</td><td><img src='{e(i['crop'])}' height='60'></td><td>{e(i['category'])}</td>"
        f"<td>{e(i['description'])}<br><small>{e(i['material'])} {e(i['brand_model'])}</small></td>"
        f"<td>{e(i['dimensions_cm']['w'])} × {e(i['dimensions_cm']['h'])} × {e(i['dimensions_cm']['d'])}</td>"
        f"<td>{e(i['status'])}</td><td>{_link(_money(i['replacement_cost']['low'], cur) + ' – ' + _money(i['replacement_cost']['high'], cur) if i['replacement_cost']['low'] is not None else '', i['replacement_cost']['url'])}"
        f"<br><small>{e(i['replacement_cost']['source'])}</small></td><td>{i['confidence']}</td>"
        f"<td><a href='{e(i['frame_ref'].split('#')[0])}'>{e(i['frame_ref'])}</a></td></tr>" for i in p["items"])
    rows_q = "".join(f"<tr><td>{e(q['ref_id'])}</td><td>{e(q['reason'])}</td></tr>" for q in p["review_queue"])
    lc = p["locale_comparison"]
    c1, c2 = lc["countries"]
    k1, k2 = lc["currencies"]
    rows_l = "".join(
        f"<tr><td>{e(x['title'])}</td><td>{_link(_money(x[c1]['amount'], k1), x[c1]['url'])}</td>"
        f"<td>{_link(_money(x[c2]['amount'], k2), x[c2]['url']) if x[c2] else 'no price found'}</td></tr>"
        for x in lc["books"])
    m = p["metrics"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Claim packet {e(p['sweep']['id'])}</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1300px}}table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}}small{{color:#555}}h2{{margin-top:32px}}</style></head><body>
<h1>Contents claim packet: home library</h1>
<p>Sweep {e(p['sweep']['id'])} · captured {e(p['sweep']['captured_at'])} · {e(p['sweep']['duration_s'])} s ·
{e(p['sweep']['country'])} / {e(cur)} · device {e(p['sweep']['device'])}</p>
<h2>Totals</h2><table>
<tr><th>Books counted</th><td>{t['book_count']} ({t['books_identified']} identified, {t['books_unidentified']} unidentified, {t['books_needs_appraisal']} need appraisal)</td></tr>
<tr><th>Shelf run</th><td>{t['shelf_run_m']} m</td></tr>
<tr><th>Books replacement cost</th><td>{_money(t['books_replacement_cost'], cur)}</td></tr>
<tr><th>Books used value</th><td>{_money(t['books_used_value'], cur)}</td></tr>
<tr><th>Other contents replacement</th><td>{_money(t['items_replacement_cost_low'], cur)} – {_money(t['items_replacement_cost_high'], cur)}</td></tr>
<tr><th>Lines excluded from totals</th><td>{t['excluded_from_totals']} (no sourced price, unidentified, or appraisal)</td></tr></table>
<p><small>Totals are summed in code from the lines below. Every price links to its listing; every line links to its saved frame.</small></p>
<h2>Room</h2><table>
<tr><th>Length × width × height</th><td>{e(r['length_m'])} × {e(r['width_m'])} × {e(r['height_m'])} m</td></tr>
<tr><th>Floor area</th><td>{e(r['floor_area_m2'])} m² ({e(r['floor_area_ft2'])} ft²)</td></tr>
<tr><th>Wall area</th><td>{e(r['wall_area_m2'])} m² ({e(r['wall_area_ft2'])} ft²) <small>{e(r['wall_area_note'])}</small></td></tr>
<tr><th>Shelved wall area</th><td>{e(r['shelved_wall_area_m2'])} m² ({e(r['shelved_wall_area_ft2'])} ft²)</td></tr>
<tr><th>Shape</th><td>{e(r['shape_method'])}</td></tr>
<tr><th>Scale</th><td>{e(r['scale_method'])}</td></tr><tr><th>Confidence</th><td>{e(r['confidence'])}</td></tr></table>
{svg}
<h2>Review queue ({len(p['review_queue'])})</h2><table><tr><th>Ref</th><th>Reason</th></tr>{rows_q}</table>
<h2>Books ({len(p['books'])})</h2><table><tr><th>ID</th><th>Shelf/pos</th><th>Spine</th><th>Status</th><th>Title / author</th>
<th>H × T cm</th><th>ID conf</th><th>Replacement</th><th>Used</th><th>Frame</th></tr>{rows_b}</table>
<h2>Other contents ({len(p['items'])})</h2><table><tr><th>ID</th><th>Crop</th><th>Category</th><th>Description</th><th>W × H × D cm</th>
<th>Status</th><th>Replacement</th><th>Conf</th><th>Frame</th></tr>{rows_i}</table>
<h2>Locale check: same books priced in {e(c1)} and {e(c2)}</h2><table><tr><th>Title</th><th>{e(c1)}</th><th>{e(c2)}</th></tr>{rows_l}</table>
<h2>Cost and latency</h2><pre>{e(json.dumps(m, indent=2))}</pre></body></html>"""
