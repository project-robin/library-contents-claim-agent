"""HTTP with retries for the catalogue and price sources. They time out and rate-limit now and then (Serper up to
~20 s on uncached queries, Google Books read timeouts); one slow request must not kill the packet."""
import time

import httpx

from pipeline import metrics


def request(method, url, tries=4, **kw):
    for attempt in range(tries):
        try:
            r = httpx.request(method, url, **kw)
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                return r
        except httpx.TransportError:
            if attempt == tries - 1:
                raise
        metrics.count(f"http_retries:{httpx.URL(url).host}")
        time.sleep(2 ** attempt)
