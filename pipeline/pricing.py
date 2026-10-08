"""Pricing stage. Every amount comes from a fetched listing (URL + retrieval time) or is left empty.
No language model is involved here."""
import base64
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


from pipeline import drex, metrics, net, settings
from pipeline.books import norm


BOOK_NEW_Q = ("Is listing {L} a new physical copy of exactly this book (same title and author, same language as the "
              "title read from the spine; any binding)? A translation into another language, an ebook, Kindle, "
              "audiobook, summary, study guide, graphic adaptation, different volume or multi-book set means no.")
BOOK_USED_Q = ("Is listing {L} a used physical copy of exactly this book (same title and author, same language as the "
               "title read from the spine)? A translation, ebook, audiobook, summary, different volume, lot or "
               "multi-book set means no.")
ITEM_BRAND_Q = "Is listing {L} a new unit of the same brand and model as the item being priced?"
ITEM_GENERIC_Q = ("Is listing {L} a new product of the same kind and similar size and material as the item being "
                  "priced, so that it is a fair replacement for insurance purposes? Accessories or parts mean no.")
APPRAISAL_Q = ("Does the evidence for book {id} indicate the copy may be rare, signed, a first or limited edition, "
               "antiquarian or otherwise worth more than an ordinary copy, so a human appraiser should value it?")

NOT_THE_BOOK = ("kindle", "ebook", "e book", "audiobook", "audio cd", "audible", "box set", "boxed set", "set of",
                "combo", "summary of", "study guide", "workbook for", "poster", "bookmark", "t shirt", "mug")
# Second-hand and import marketplaces that list on Google Shopping. Their offers were being taken as "new" prices
# (Biblio's ₹492 hardcover over amazon.in's ₹284 paperback) and made re-runs swing up to 25x.
USED_SELLERS = ("bookchor", "kitabay", "biblio", "better world books", "abebooks", "thriftbooks", "world of books",
                "wob", "alibris", "half price books", "awesomebooks", "secondhand", "second hand", "used book",
                "pre-owned", "preowned", "ebay",
                "pressreader", "scribd", "kobo", "google play", "audible")  # digital / subscription, not a copy
USED_WORDS = (" used ", " pre owned ", " preowned ", " second hand ", " secondhand ", " old book ")
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


def not_the_book(listing_title):
    """Cheap whole-word prefilter before DREX (saves tokens). 'mug' must not reject 'The Smuggler'."""
    t = f" {norm(listing_title)} "
    return any(f" {w} " in t for w in NOT_THE_BOOK)


def used_listing(retailer, listing_title):
    r, t = retailer.lower(), f" {norm(listing_title)} "
    return any(s in r for s in USED_SELLERS) or any(w in t for w in USED_WORDS)


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
            r = net.request("POST", "https://google.serper.dev/shopping", timeout=45,  # uncached queries take up to ~20 s
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
                t = net.request("POST", "https://api.ebay.com/identity/v1/oauth2/token", timeout=20,
                               headers={"Authorization": f"Basic {auth}"},
                               data={"grant_type": "client_credentials",
                                     "scope": "https://api.ebay.com/oauth/api_scope"})
                t.raise_for_status()
                self._ebay_token = t.json()["access_token"]
            r = net.request("GET", "https://api.ebay.com/buy/browse/v1/item_summary/search", timeout=20,
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
            r = net.request("GET", "https://api.frankfurter.dev/v1/latest", params={"base": base, "symbols": to}, timeout=15)
            r.raise_for_status()
            j = r.json()
            self.cache[key] = {"rate": j["rates"][to], "date": j["date"], "retrieved_at": now(),
                               "url": str(r.url), "source": "Frankfurter (European Central Bank reference rates)"}
        return self.cache[key]

    # ---- listing match: DREX decides whether each listing is the thing being priced ----
    def matching(self, subject, offers, question):
        """Keep offers whose listing DREX judges to match `subject` with P >= LISTING_MATCH_MIN. Cached for reruns."""
        if not offers:
            return []
        key = f"drex|{question[:40]}|{json.dumps(subject, sort_keys=True)}|" + "|".join(o["title"] for o in offers)
        if key not in self.cache:
            state = {"item_being_priced": subject, "listings": {f"L{i}": o["title"] for i, o in enumerate(offers)}}
            self.cache[key] = drex.ask(state, {f"L{i}": question.format(L=f"L{i}") for i in range(len(offers))})
        p = self.cache[key]
        return [o | {"match_p": round(p.get(f"L{i}", 0), 3)} for i, o in enumerate(offers)
                if p.get(f"L{i}", 0) >= settings.LISTING_MATCH_MIN]

    # ---- offers ----
    def book_offers(self, title, author, country):
        """Union of three queries. Google Shopping returns a different handful of listings on every call; one query
        left most books on 1-2 matched offers and two fresh runs agreed within 25% on 9/12 books (3 queries: 11/12)."""
        loc = settings.LOCALES[country]
        first = author.split(",")[0].strip()
        last = first.split()[-1] if first else ""
        seen, offers, at = set(), [], ""
        for q in (f"{title} {last} book", f"{title} {first} paperback", f"{title} {first}"):
            res = self.shopping(q.strip(), loc["gl"])
            at = at or res["retrieved_at"]
            for x in res["results"]:
                link, lt = x.get("link"), x.get("title", "")
                if not link or link in seen or not_the_book(lt) or used_listing(x.get("source", ""), lt):
                    continue
                if (a := parse_price(x.get("price"), loc["symbols"])):
                    seen.add(link)
                    offers.append({"amount": a, "retailer": x.get("source", ""), "url": link, "title": lt})
        return self.matching({"title": title, "author": author}, offers, BOOK_NEW_Q), at

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
                  for x in res["results"] if x.get("price") and x.get("itemWebUrl")
                  and not not_the_book(x.get("title", ""))]
        offers = self.matching({"title": title, "author": author}, offers, BOOK_USED_Q)
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
        offers = [{"amount": a, "retailer": x.get("source", ""), "url": x.get("link", ""), "title": x.get("title", "")}
                  for x in res["results"] if x.get("link") and (a := parse_price(x.get("price"), loc["symbols"]))]
        subject = {k: item[k] for k in ("category", "description", "material", "brand_model", "dimensions_cm")}
        offers = self.matching(subject, offers, ITEM_BRAND_Q if item["brand_model"] else ITEM_GENERIC_Q)
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
    evidence = {b["id"]: {"title": b["title"], "visible_marks": b["marks"], "user_said": b["notes"]}
                for b in books if not b["excluded"] and (b["marks"] or b["notes"])}
    appraisal = drex.ask({"books": evidence}, {k: APPRAISAL_Q.format(id=k) for k in evidence}) if evidence else {}
    try:
        def price_book(b):
            p = {"status": "identified", "reasons": []}
            if appraisal.get(b["id"], 0) >= settings.APPRAISAL_MIN:
                ev = ", ".join(b["marks"] + [f"user: {n}" for n in b["notes"]])
                p.update(status="needs_appraisal", appraisal_p=round(appraisal[b["id"]], 3),
                         reasons=[f"possible rare/signed/special copy (P={appraisal[b['id']]:.2f}; {ev}): not auto-priced"])
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
                    if new and new["offers_considered"] < settings.MIN_PRICE_LISTINGS:
                        p["reasons"].append(f"replacement price rests on {new['offers_considered']} matching "
                                            "listing(s): may not be the market price")
            return p

        todo = [b for b in books if not b["excluded"]]
        with ThreadPoolExecutor(16) as ex:  # Serper: 1-3 s typical, 20-30 s tails and rare 500s; overlap them
            for b, p in zip(todo, ex.map(price_book, todo)):
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
