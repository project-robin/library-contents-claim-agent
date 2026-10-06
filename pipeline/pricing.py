"""Pricing stage. Every amount comes from a fetched listing (URL + retrieval time) or is left empty.
No language model is involved here."""
import base64
import json
import re
import statistics
from datetime import datetime, timezone

import httpx

from pipeline import metrics, settings
from pipeline.books import norm

NOT_THE_BOOK = ("kindle", "ebook", "e book", "audiobook", "audio cd", "audible", "box set", "boxed set", "set of",
                "combo", "summary of", "study guide", "workbook for", "poster", "bookmark", "t shirt", "mug")
FOREIGN_DOLLAR = ("A$", "C$", "CA$", "AU$", "NZ$", "HK$", "S$", "SG$", "R$", "MX$")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_price(s, symbols):
    """'₹1,29,999.00' -> 129999.0. None if the string is not in this locale's currency."""
    if not s or not any(sym in s for sym in symbols):
        return None
    if "$" in symbols and any(f in s for f in FOREIGN_DOLLAR):
        return None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    return float(m.group().replace(",", "")) if m else None


def title_matches(want, got):
    """All significant words of the wanted title appear in the listing title, and it is not an ebook/set/etc."""
    g = norm(got)
    if any(x in g for x in NOT_THE_BOOK):
        return False
    words = [w for w in norm(want).split() if len(w) > 2] or norm(want).split()
    return bool(words) and all(w in g.split() for w in words)


def lower_median(offers):
    """A real offer at the median (not an average of two), so the cited URL is the price used."""
    s = sorted(offers, key=lambda o: o["amount"])
    return s[(len(s) - 1) // 2]


class Pricer:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._ebay_token = None

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache))

    # ---- sources ----
    def shopping(self, q, gl):
        key = f"serper|{gl}|{q}"
        if key not in self.cache:
            if not settings.SERPER_API_KEY:
                return {"retrieved_at": "", "results": []}
            r = httpx.post("https://google.serper.dev/shopping", timeout=20,
                           headers={"X-API-KEY": settings.SERPER_API_KEY},
                           json={"q": q, "gl": gl, "hl": "en", "num": 20})
            metrics.count("serper_queries")
            r.raise_for_status()
            self.cache[key] = {"retrieved_at": now(), "results": r.json().get("shopping", [])}
        return self.cache[key]

    def ebay_used(self, q, market):
        key = f"ebay|{market}|{q}"
        if key not in self.cache:
            if not (settings.EBAY_CLIENT_ID and settings.EBAY_CLIENT_SECRET):
                return {"retrieved_at": "", "results": []}
            if not self._ebay_token:
                auth = base64.b64encode(f"{settings.EBAY_CLIENT_ID}:{settings.EBAY_CLIENT_SECRET}".encode()).decode()
                t = httpx.post("https://api.ebay.com/identity/v1/oauth2/token", timeout=20,
                               headers={"Authorization": f"Basic {auth}"},
                               data={"grant_type": "client_credentials",
                                     "scope": "https://api.ebay.com/oauth/api_scope"})
                t.raise_for_status()
                self._ebay_token = t.json()["access_token"]
            r = httpx.get("https://api.ebay.com/buy/browse/v1/item_summary/search", timeout=20,
                          headers={"Authorization": f"Bearer {self._ebay_token}", "X-EBAY-C-MARKETPLACE-ID": market},
                          params={"q": q, "category_ids": "267", "limit": 30,
                                  "filter": "conditions:{USED},buyingOptions:{FIXED_PRICE}"})
            metrics.count("ebay_queries")
            r.raise_for_status()
            self.cache[key] = {"retrieved_at": now(), "results": r.json().get("itemSummaries", [])}
        return self.cache[key]

    def fx(self, base, to):
        key = f"fx|{base}|{to}"
        if key not in self.cache:
            r = httpx.get("https://api.frankfurter.dev/v1/latest", params={"base": base, "symbols": to}, timeout=15)
            r.raise_for_status()
            j = r.json()
            self.cache[key] = {"rate": j["rates"][to], "date": j["date"], "retrieved_at": now(),
                               "url": str(r.url), "source": "Frankfurter (European Central Bank reference rates)"}
        return self.cache[key]

    # ---- offers ----
    def book_offers(self, title, author, country):
        loc = settings.LOCALES[country]
        last = author.split(",")[0].split()[-1] if author else ""
        res = self.shopping(f"{title} {last} book".strip(), loc["gl"])
        offers = [{"amount": a, "retailer": x.get("source", ""), "url": x.get("link", ""), "title": x.get("title", "")}
                  for x in res["results"]
                  if title_matches(title, x.get("title", "")) and (a := parse_price(x.get("price"), loc["symbols"]))]
        return offers, res["retrieved_at"]

    def convert(self, amount, base, to):
        if base == to:
            return amount, None
        fx = self.fx(base, to)
        return round(amount * fx["rate"], 2), fx

    def book_new(self, title, author, country):
        loc = settings.LOCALES[country]
        offers, at = self.book_offers(title, author, country)
        if offers:
            o = lower_median(offers)
            return {"amount": o["amount"], "source": f"Google Shopping via Serper: {o['retailer']}", "url": o["url"],
                    "retrieved_at": at, "converted": False, "offers_considered": len(offers),
                    "listing_title": o["title"]}
        for other in ("US", "GB"):
            if other == country:
                continue
            offers, at = self.book_offers(title, author, other)
            if offers:
                o = lower_median(offers)
                amt, fx = self.convert(o["amount"], settings.LOCALES[other]["currency"], loc["currency"])
                return {"amount": amt, "source": f"Google Shopping via Serper ({other}): {o['retailer']}",
                        "url": o["url"], "retrieved_at": at, "converted": True, "offers_considered": len(offers),
                        "listing_title": o["title"],
                        "converted_from": {"amount": o["amount"], "currency": settings.LOCALES[other]["currency"],
                                           "fx": fx}}
        return None

    def book_used(self, title, author, country):
        loc = settings.LOCALES[country]
        market = loc["ebay"] or settings.USED_FALLBACK_MARKET
        last = author.split(",")[0].split()[-1] if author else ""
        res = self.ebay_used(f"{title} {last}".strip(), market)
        offers = [{"amount": float(x["price"]["value"]), "currency": x["price"]["currency"],
                   "url": x.get("itemWebUrl", ""), "title": x.get("title", ""), "condition": x.get("condition", "")}
                  for x in res["results"] if x.get("price") and title_matches(title, x.get("title", ""))]
        if not offers:
            return None
        o = lower_median(offers)
        amt, fx = self.convert(o["amount"], o["currency"], loc["currency"])
        out = {"amount": amt, "source": f"eBay {market} used fixed-price listings", "url": o["url"],
               "retrieved_at": res["retrieved_at"], "condition_assumed": f"used, {o['condition'] or 'as listed'} "
               f"(median of {len(offers)} matching used listings)", "converted": fx is not None,
               "listing_title": o["title"]}
        if fx:
            out["converted_from"] = {"amount": o["amount"], "currency": o["currency"], "fx": fx}
        return out

    def item_range(self, item, country):
        loc = settings.LOCALES[country]
        is_art = item["category"] == "art"
        q = item["brand_model"] or f"{item['material']} {item['description']}".strip()
        if is_art:
            q = f"{item['description']} framed print"
        res = self.shopping(q, loc["gl"])
        brand_words = norm(item["brand_model"]).split()
        offers = [{"amount": a, "retailer": x.get("source", ""), "url": x.get("link", "")}
                  for x in res["results"]
                  if (a := parse_price(x.get("price"), loc["symbols"]))
                  and (not brand_words or any(w in norm(x.get("title", "")).split() for w in brand_words))]
        if not offers:
            return None, q
        amts = sorted(o["amount"] for o in offers)
        if item["brand_model"]:
            o = lower_median(offers)
            lo = hi = o["amount"]
            urls = [o["url"]]
        else:
            if len(amts) >= 3:
                q1, _, q3 = statistics.quantiles(amts, n=4)
                lo, hi = round(q1, 2), round(q3, 2)
            else:
                lo, hi = amts[0], amts[-1]
            urls = [o["url"] for o in offers if lo <= o["amount"] <= hi][:5]
        return {"low": lo, "high": hi, "source": f"Google Shopping via Serper, query '{q}' ({len(offers)} listings)",
                "url": urls[0] if urls else "", "urls": urls, "retrieved_at": res["retrieved_at"]}, q


def run(sweep_dir, books, items, country):
    loc = settings.LOCALES[country]
    pr = Pricer(sweep_dir / "price_cache.json")
    thr = loc["appraisal_threshold"]
    out_books, out_items = {}, {}
    try:
        for b in books:
            if b["excluded"]:
                continue
            p = {"status": "identified", "reasons": []}
            if b["appraisal_reasons"]:
                p.update(status="needs_appraisal", reasons=[f"possible rare/signed/special: {', '.join(b['appraisal_reasons'])}"])
            elif not b["title"]:
                p["status"] = "unidentified"
            else:
                new, used = pr.book_new(b["title"], b["author"], country), pr.book_used(b["title"], b["author"], country)
                over = [x for x in (new, used) if x and x["amount"] > thr]
                if over:
                    p.update(status="needs_appraisal",
                             reasons=[f"retrieved price above appraisal threshold {thr} {loc['currency']}: not auto-priced"])
                else:
                    p.update(replacement_cost=new, used_value=used)
            out_books[b["id"]] = p

        for it in items:
            p = {"status": "range", "reasons": []}
            if it["category"] == "art" and it["is_print"] is not True:
                p.update(status="needs_appraisal", reasons=["art defaults to appraisal unless the user confirms a print"])
            else:
                rng, q = pr.item_range(it, country)
                p["query"] = q
                if rng:
                    p.update(status="priced" if it["brand_model"] else "range", replacement_cost=rng)
                else:
                    p["reasons"].append(f"no retrievable price for query '{q}'")
            out_items[it["id"]] = p

        comparison = []
        other = settings.COMPARISON_COUNTRY if settings.COMPARISON_COUNTRY != country else "GB"
        for b in [b for b in books if out_books.get(b["id"], {}).get("replacement_cost")][:10]:
            comparison.append({"id": b["id"], "title": b["title"],
                               country: out_books[b["id"]]["replacement_cost"],
                               other: pr.book_new(b["title"], b["author"], other)})
    finally:
        pr.save()
    return {"books": out_books, "items": out_items,
            "locale_comparison": {"countries": [country, other], "currencies": [loc["currency"], settings.LOCALES[other]["currency"]],
                                  "books": comparison}}
