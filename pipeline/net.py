"""HTTP with retries for the catalogue and price sources. They time out and rate-limit now and then (Serper up to
~20 s on uncached queries, Google Books read timeouts); one slow request must not kill the packet."""
import time

import httpx

from pipeline import metrics


def request(method, url, **kw):
    for attempt in range(4):
        try:
            r = httpx.request(method, url, **kw)
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                return r
        except httpx.TransportError:
            if attempt == 3:
                raise
        metrics.count(f"http_retries:{httpx.URL(url).host}")
        time.sleep(2 ** attempt)
