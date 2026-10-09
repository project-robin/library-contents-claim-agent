"""Pricing stage. Every amount comes from a fetched listing (URL + retrieval time) or is left empty.
No language model is involved here."""
import base64
import json
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


import httpx

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
# They are the used-price pool instead (India has no used-book API).
USED_SELLERS = ("bookchor", "kitabay", "biblio", "better world books", "abebooks", "thriftbooks", "world of books",
                "wob", "alibris", "half price books", "awesomebooks", "secondhand", "second hand", "used book",
                "pre-owned", "preowned", "ebay")
DIGITAL_SELLERS = ("pressreader", "scribd", "kobo", "google play", "audible")  # subscription / digital, not a copy
USED_WORDS = (" used ", " pre owned ", " preowned ", " second hand ", " secondhand ", " old book ")
# A listing in another language is a translation, whatever DREX says (it passed 'Think Like a Monk (Kannada)')
LANGUAGES = ("hindi", "kannada", "tamil", "telugu", "marathi", "bengali", "bangla", "malayalam", "gujarati", "urdu",
             "punjabi", "odia", "assamese", "sanskrit", "spanish", "french", "german", "italian", "portuguese",
             "japanese", "chinese", "korean", "russian", "arabic")
_LANG = "|".join(LANGUAGES)
# marked as a language, not a subtitle word ('Ikigai: The Japanese Secret to a Long and Happy Life' is English)
LANGUAGE_MARK = re.compile(rf"[(\[]\s*(?:{_LANG})\b|\b(?:{_LANG})\s+(?:edition|version|translation)\b|"
                           rf"\b(?:in|into)\s+(?:{_LANG})\b")
# Devanagari ... Sinhala, Cyrillic, Arabic, CJK, Hangul: 'Ikigai (इकिगाई)' is the Hindi edition
OTHER_SCRIPT = re.compile("[\u0400-\u04ff\u0600-\u06ff\u0900-\u0dff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
STOP = {"the", "and", "for", "with", "from", "book", "books", "novel", "paperback", "hardcover", "edition", "english",
        "new", "original", "author", "by"}
FOREIGN_WORDS = {"libro", "livre", "buch", "kniha", "beletrie", "novela"}  # 'Libro Will Grayson Will Grayson'
SET_WORDS = {"set", "sets", "collection", "combo", "bundle", "lot"}  # 'The Essential John Green Collection 4 Books Set'
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


def same_book_title(title, listing_title, author=""):
    """Most of the title's words appear in the listing, and the listing is not a translation or a set. DREX alone
    passed 'Flyte By Mark Zug' as a used copy of 'The Oath of the Vayuputras'. A listing may drop the subtitle
    ('Japonisme | Erin Niimi Longhurst') if nothing but the author is left over; 'Tom Gates #24: Pesky Pets And
    Parties' for 'Tom Gates: Family, Friends and Furry Creatures' is another book in the series."""
    t, lt = set(norm(title).split()), set(norm(listing_title).split())
    if LANGUAGE_MARK.search(listing_title.lower()) and not LANGUAGE_MARK.search(title.lower()):
        return False
    if OTHER_SCRIPT.search(listing_title) and not OTHER_SCRIPT.search(title):
        return False
    if (lt & FOREIGN_WORDS) - t or (lt & SET_WORDS) - t:
        return False
    words = {w for w in t if len(w) >= 3 and w not in STOP}
    if not words or len(words & lt) >= 0.6 * len(words):
        return True
    main = {w for w in norm(re.split(r"[:(\[|]| - ", title)[0]).split() if w not in STOP}
    rest = {w for w in lt - main - set(norm(author).split()) if len(w) >= 3 and w not in STOP}
    return bool(main) and main <= lt and len(rest) <= 1


def used_listing(retailer, listing_title):
    """A second-hand copy: excluded from new prices, kept for the used price."""
    r, t = retailer.lower(), f" {norm(listing_title)} "
    return any(s in r for s in USED_SELLERS) or any(w in t for w in USED_WORDS)


def digital_listing(retailer):
    return any(s in retailer.lower() for s in DIGITAL_SELLERS)


def lower_median(offers):
    """A real offer at the median (not an average of two), so the cited URL is the price used."""
    s = sorted(offers, key=lambda o: o["amount"])
    return s[(len(s) - 1) // 2]


# median 2-3 s, but 1 in 20 runs 15-45 s; a retry after a 12 s cutoff is usually quick. Two tries only: four
# (12 s each plus back-off) once held a whole sweep's pricing for 55 s; a book still has its other two queries
SERPER_TIMEOUT_S = 12
SERPER_PARALLEL = 8  # requests in flight: 48 at once drew 111 HTTP 429s for 106 answers, and their back-off waits
_serper_slots = threading.BoundedSemaphore(SERPER_PARALLEL)


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
            with _serper_slots:
                r = net.request("POST", "https://google.serper.dev/shopping", timeout=SERPER_TIMEOUT_S, tries=2,
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
    def listings(self, title, author, country):
        """Union of three queries, split into new and second-hand listings (unmatched). Google Shopping returns a
        different handful of listings on every call; one query left most books on 1-2 matched offers and two fresh
        runs agreed within 25% on 9/12 books (3 queries: 11/12). Cached, so the used price costs no extra query."""
        loc = settings.LOCALES[country]
        first = author.split(",")[0].strip()
        last = first.split()[-1] if first else ""
        seen, new, used, at = set(), [], [], ""
        qs = [q.strip() for q in (f"{title} {last} book", f"{title} {first} paperback", f"{title} {first}")]

        def one(q):  # in parallel: one at a time, a book whose queries each hit a 15-40 s Serper tail took 95 s
            try:
                return self.shopping(q, loc["gl"])
            except httpx.HTTPError as e:
                return e
        with ThreadPoolExecutor(3) as ex:
            got = list(ex.map(one, qs))
        if all(isinstance(r, Exception) for r in got):
            raise got[0]
        for res in (r for r in got if not isinstance(r, Exception)):  # a failed query among three: price on the rest
            at = at or res["retrieved_at"]
            for x in res["results"]:
                link, lt = x.get("link"), x.get("title", "")
                seller = x.get("source", "")
                if not link or link in seen or not_the_book(lt) or digital_listing(seller):
                    continue
                if (a := parse_price(x.get("price"), loc["symbols"])):
                    seen.add(link)
                    o = {"amount": a, "retailer": seller, "url": link, "title": lt}
                    (used if used_listing(seller, lt) else new).append(o)
        return new, used, at

    def book_offers(self, title, author, country):
        new, _, at = self.listings(title, author, country)
        new = [o for o in new if same_book_title(title, o["title"], author)]
        return self.matching({"title": title, "author": author}, new, BOOK_NEW_Q), at

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
        # both fallback markets at once, US preferred: one after the other, a book with no local listing held a live
        # sweep's pricing for 108 s on Serper's slow tails
        def offers_in(other):
            try:
                return self.book_offers(title, author, other)
            except httpx.HTTPError as e:
                return e
        others = [o for o in ("US", "GB") if o != country]
        with ThreadPoolExecutor(len(others)) as ex:
            got = list(ex.map(offers_in, others))
        if all(isinstance(g, Exception) for g in got):
            raise got[0]
        for other, res in zip(others, got):
            if isinstance(res, Exception):
                continue
            offers, at = res
            if offers:
                o = lower_median(offers)
                amt, fx = self.convert(o["amount"], settings.LOCALES[other]["currency"], loc["currency"])
                return {"amount": amt, "source": f"Google Shopping via Serper ({other}): {o['retailer']}",
                        "url": o["url"], "retrieved_at": at, "converted": True, "offers_considered": len(offers),
                        "listing_title": o["title"],
                        "converted_from": {"amount": o["amount"], "currency": settings.LOCALES[other]["currency"],
                                           "fx": fx}}
        return None

    def book_used(self, title, author, country, new=None):
        """Second-hand sellers on local Google Shopping first, then eBay when keys are set. A used offer above the
        new price is a collector or import listing, not what an ordinary used copy fetches: dropped."""
        loc = settings.LOCALES[country]
        _, raw, at = self.listings(title, author, country)
        raw = [o for o in raw if same_book_title(title, o["title"], author)]
        # DREX sees the seller: a listing title alone does not say whether the copy is used
        offers = self.matching({"title": title, "author": author},
                               [o | {"listing": o["title"], "title": f"{o['title']} (sold by {o['retailer']}, second-hand seller)"}
                                for o in raw],
                               BOOK_USED_Q)
        offers = [o for o in offers if not new or o["amount"] <= new["amount"]]
        if offers:
            o = lower_median(offers)
            return {"amount": o["amount"], "source": f"Google Shopping via Serper: {o['retailer']} (second-hand seller)",
                    "url": o["url"], "retrieved_at": at,
                    "condition_assumed": f"used, as listed by a second-hand seller (median of {len(offers)} matching "
                                         "used listings)",
                    "converted": False, "listing_title": o["listing"]}
        market = loc["ebay"] or settings.USED_FALLBACK_MARKET
        last = author.split(",")[0].split()[-1] if author else ""
        res = self.ebay_used(f"{title} {last}".strip(), market)
        offers = [{"amount": float(x["price"]["value"]), "currency": x["price"]["currency"],
                   "url": x.get("itemWebUrl", ""), "title": x.get("title", ""), "condition": x.get("condition", "")}
                  for x in res["results"] if x.get("price") and x.get("itemWebUrl")
                  and not not_the_book(x.get("title", "")) and same_book_title(title, x.get("title", ""), author)]
        offers = self.matching({"title": title, "author": author}, offers, BOOK_USED_Q)
        if not offers:
            return None
        o = lower_median(offers)
        amt, fx = self.convert(o["amount"], o["currency"], loc["currency"])
        if new and amt > new["amount"]:
            return None
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
                try:
                    new = pr.book_new(b["title"], b["author"], country)
                    used = pr.book_used(b["title"], b["author"], country, new)
                except httpx.HTTPError as e:  # Serper 504s outlast the retries: an unpriced line, not a dead packet
                    metrics.count("price_failures")
                    p["reasons"].append(f"price lookup failed ({type(e).__name__}): not priced")
                    return p
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

        def price_item(it):
            p = {"status": "range", "reasons": []}
            if it["category"] == "art" and it["is_print"] is not True:
                p.update(status="needs_appraisal", reasons=["art defaults to appraisal unless the user confirms a print"])
                return p
            try:
                rng, q = pr.item_range(it, country)
            except httpx.HTTPError as e:
                metrics.count("price_failures")
                rng, q = None, ""
                p["reasons"].append(f"price lookup failed ({type(e).__name__}): not priced")
            p["query"] = q
            if rng:
                p.update(status="priced" if it["brand_model"] else "range", replacement_cost=rng)
            elif q:
                p["reasons"].append(f"no retrievable price for query '{q}'")
            return p

        other = settings.COMPARISON_COUNTRY if settings.COMPARISON_COUNTRY != country else "GB"

        def compare(b, local):
            try:
                return {"id": b["id"], "title": b["title"], country: local,
                        other: pr.book_new(b["title"], b["author"], other)}
            except httpx.HTTPError:
                metrics.count("price_failures")

        # one pool for all lookups (Serper: 1-3 s typical, 15-45 s tails and rare 500s; overlap them). The locale
        # comparison starts as each book is priced (the first 10 to finish): started after the slowest book, it added
        # 38 s to a live sweep's 145 s of pricing
        todo = [b for b in books if not b["excluded"]]
        compared, slot = [], threading.Lock()
        with ThreadPoolExecutor(16) as ex:
            def price_then_compare(b):
                p = price_book(b)
                if p.get("replacement_cost"):
                    with slot:
                        if len(compared) < 10:
                            compared.append(ex.submit(compare, b, p["replacement_cost"]))
                return p
            item_prices = ex.map(price_item, items)  # submitted now, runs alongside the books
            out_books.update(zip((b["id"] for b in todo), ex.map(price_then_compare, todo)))
            order = {b["id"]: i for i, b in enumerate(books)}
            comparison = sorted((c for c in (f.result() for f in compared) if c), key=lambda c: order[c["id"]])
            out_items.update(zip((it["id"] for it in items), item_prices))
    finally:
        pr.save()
    return {"books": out_books, "items": out_items,
            "locale_comparison": {"countries": [country, other], "currencies": [loc["currency"], settings.LOCALES[other]["currency"]],
                                  "books": comparison}}
