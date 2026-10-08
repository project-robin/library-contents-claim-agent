"""Check every outside service with one small call before a sweep, so a spent quota does not waste a take.

    uv run python -m eval.preflight

Free tiers run out per day (Gemini Flash 20/day, Flash-Lite 500/day, Google Books per project): a sweep started on a
spent quota builds a packet with every title unidentified and every check 'unsure'. One call per service, about 10 s.
"""
import time

import httpx

from pipeline import net, settings


def gemini(model, key):
    from pipeline import vision
    vision.client(key).models.generate_content(model=model, contents="Reply with the word ok.")


def gbooks():
    key = settings.GOOGLE_BOOKS_API_KEY or settings.GEMINI_API_KEY
    r = httpx.get("https://www.googleapis.com/books/v1/volumes", timeout=20,
                  params={"q": "intitle:Killing Floor", "maxResults": 1} | ({"key": key} if key else {}))
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.json().get('error', {}).get('message', '')[:120]}")


def openlibrary():
    httpx.get("https://openlibrary.org/search.json", params={"title": "Killing Floor", "limit": 1},
              timeout=20).raise_for_status()


def serper():
    if not settings.SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY empty")
    net.request("POST", "https://google.serper.dev/shopping", timeout=45, headers={"X-API-KEY": settings.SERPER_API_KEY},
                json={"q": "Killing Floor Lee Child", "gl": "in", "num": 1}).raise_for_status()


def sam():
    from pipeline import sam as sam_client
    if not sam_client.available():
        raise RuntimeError("META_API_KEY empty: spines will come from Gemini boxes")


def ocr():
    from gpu.modal_ocr import remote
    remote().ping.remote()


# every key in turn (shown by its last 4 characters): the pipeline moves to the next key when one's daily quota runs out
CHECKS = [(f"Gemini {label} ({model}) key …{key[-4:]}", lambda model=model, key=key: gemini(model, key))
          for key in settings.gemini_keys()
          for label, model in (("reads", settings.VISION_MODEL), ("fallback", settings.VISION_FALLBACK_MODEL),
                               ("checker", settings.CHECKER_MODEL))]
CHECKS += [("Google Books", gbooks), ("Open Library", openlibrary), ("Serper shopping", serper),
          ("SAM 3.1 key", sam), ("PP-OCRv5 on Modal (wakes the container)", ocr)]


if __name__ == "__main__":
    bad, gemini_ok = 0, set()
    for name, fn in CHECKS:
        t = time.time()
        try:
            fn()
            print(f"  ok    {name}  ({time.time() - t:.1f} s)")
            gemini_ok.add(name.split(" key ")[0])
        except Exception as e:  # noqa: BLE001 - report every failure, keep checking the rest
            bad += 1 if " key …" not in name else 0  # a spent key is fine while another key has quota
            msg = str(e)
            wait = msg.split("Please retry in ")[1].split(".")[0] if "Please retry in " in msg else ""
            print(f"  FAIL  {name}: {msg.splitlines()[0][:110]}" + (f"  -> retry in {wait}" if wait else ""))
    bad += len({n.split(" key ")[0] for n, _ in CHECKS if " key …" in n} - gemini_ok)  # no key left for a model
    print("\nAll services answer: start the sweep." if not bad else
          f"\n{bad} service(s) failed. A spent daily quota resets within a day (Gemini's 429 says 'retry in ...'); "
          "a sweep now gets blank titles and 'unsure' checks.")
