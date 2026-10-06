# Library Contents Claim Agent

A policyholder walks their home library once with a phone. A live voice agent directs the sweep. When the sweep
ends, separate pipeline stages count, read, measure and price every spine and object, and measure the room. The
output is a `claim_packet.json` plus an HTML report. Every number traces back to a saved frame and a price URL.

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Results against ground truth: [docs/RESULTS.md](docs/RESULTS.md)
- Failure log: [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md)
- Reference app (kept / changed / thrown away): [docs/REFERENCE_NOTES.md](docs/REFERENCE_NOTES.md)

## Setup (about 10 minutes)

You need Python tooling [`uv`](https://docs.astral.sh/uv/), an Android phone with Chrome on the same Wi-Fi, and
accounts for Gemini, Serper and Modal (eBay is optional).

```bash
git clone <repo> && cd <repo>
uv sync                                   # Python 3.12 + deps; no torch locally
cp .env.example .env                      # add GEMINI_API_KEY, SERPER_API_KEY, DREX_API_KEY (+ eBay)
uv run modal token new                    # once, opens browser
uv run modal deploy gpu/modal_recon.py    # GPU reconstruction service (first build ~5 min)
uv run python -m pipeline.marker          # -> assets/aruco_markers_A4.pdf
```

Print the marker PDF at **100% / actual size**. Measure the black square with a ruler and set `MARKER_SIZE_CM` in
`.env`, because printers rescale.

## Run a sweep

```bash
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
# phone cameras need HTTPS -> in a second terminal, a no-account tunnel:
docker run --rm --network host cloudflare/cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://….trycloudflare.com` URL in Chrome on the phone and tap **Start sweep**. The agent
confirms your country, then asks you to lean the marker against a shelf and walk the room once. Say "done" or tap
**End sweep**. The report link appears when the packet is ready, and everything is saved under `sweeps/<id>/`.

## Rerun or score offline

```bash
uv run python -m pipeline.run sweeps/<id>                 # full pipeline from saved evidence
uv run python -m pipeline.run sweeps/<id> --from pricing  # any stage: keyframes reads recon geometry books checker items pricing packet
uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/ --md
uv run pytest                                             # offline tests incl. a synthetic room end-to-end
```

## What is used

| Purpose | Tool |
|---|---|
| Live voice + camera agent | Gemini Live (`gemini-3.1-flash-live-preview`) via `google-genai` |
| Spine/item detection + transcription | Gemini Flash, JSON schema output |
| Independent identity check | Gemini Pro (different model; same vendor, noted as a limit) |
| Camera poses + depth | VGGT-1B (Meta, pretrained) on a Modal L40S GPU |
| Metric scale | Printed ArUco marker (OpenCV), with a cross-check against standard book formats |
| Book identity | Google Books API, Open Library |
| New prices | Google Shopping via Serper.dev (per-country `gl`) |
| Used prices | eBay Browse API (used, fixed price) |
| FX | Frankfurter (ECB reference rates) |
| Backend/UI | FastAPI, one HTML page, plain JS |
| Coding | Built with Claude Code (AI pair programmer) |

No model is trained or fine-tuned. No catalogue, price list or room dimensions are hardcoded.

## Assumptions (where the brief is ambiguous)

- **Wall area** is gross (perimeter × height). Doors and windows are not subtracted.
- **Replacement cost** is the lower-median matched new listing in the local market (a real offer, not an average).
- **Used value** is the lower-median matched used listing. India has no used-book API, so Indian used prices are
  converted from eBay US and labelled `converted`.
- **`excluded_from_totals`** counts lines (books + items) that add nothing to replacement totals: no sourced price,
  unidentified, or needs appraisal.
- **Appraisal triggers**: visible or stated signed / first edition / limited / leather / gilt / old binding, or a
  retrieved price above the threshold (₹5,000 / $75 / £60). These lines are not priced. Publication year alone is
  not a trigger, because the catalogue year belongs to the work, not the copy.
- **Shelf** = unit letter + row from the top (`B-3`). **Position** counts along the shelf.
- **Shelved wall area** = sum of the front faces (width × height) of the detected shelving units.
