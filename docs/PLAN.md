# Build Plan — Library Contents Claim Agent

Approved 2026-10-06. Deadline ≈ 2026-10-09 02:30 IST (72h from brief at 02:38 on Oct 6).
Brief: `../Take-Home Assignment_ Library Contents Claim Agent.md`

## Revision 2 (2026-10-06, after tool research + live tests)

| Change | Evidence | Status |
|---|---|---|
| Models → `gemini-3.8-live` (stable Live default) + `gemini-3.8-flash`, thinking `low`, fallback `gemini-flash-latest` | 3.1-flash-live is now "legacy"; 3.8 Flash read 22/22 spines incl. a flat book on a test shelf; frequent 503 "high demand" → retries + fallback | done |
| **DREX** (Nace.AI decision model) for every judgement: identity P(same work), listing match, appraisal | live: Sapiens listing check 0.99 vs Kindle 0.002 / Graphic Vol.1 0.013 (old word-rule wrongly accepted it); identities 0.92–0.998, ambiguous "Erzählungen 2" 0.518 → review | done |
| **pydantic-settings** typed config + `require()` fail-fast; **pydantic output contract** validated before writing | missing key now fails at startup; unsourced price / dropped field / priced appraisal item fail validation | done |
| **MapAnything (Apache)** added next to VGGT on Modal; `RECON_MODEL` picks; metric output = independent cross-check of marker, fallback when marker unseen | VGGT has no metric output + non-commercial licence; MapAnything better on textureless surfaces | code done, spike pending (needs Modal) |
| **SAM 3 / 3.1** → *not* in core path; optional eval cross-check via free HF Space | on a 22-spine shelf: Gemini boxes vs SAM masks median IoU 0.944, width diff 3.8%; SAM missed the flat book; free Space 80 s/image | decided |
| Catalogue search strips volume markers; Books API reuses Gemini key | anonymous Google Books quota exhausted (429); "In Nacht und Eis II" 0 hits → stripped 12 hits | done (user: enable Books API on the Gemini project) |
| Free HF route for MapAnything/VGGT | no inference providers; Spaces expose only a GLB viewer, not depth/poses → Modal free credits instead | decided |

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Web stack | FastAPI + one static `index.html` + vanilla JS. **No Next.js** | Pipeline is Python; reference is FastAPI+JS; polish/deploy unscored; Vercel can't hold Live websocket; one-command setup |
| Package mgr | `uv`, Python 3.12 | Fast, reproducible, stranger setup < 15 min |
| Live agent | Gemini Live (native audio), relayed through backend | Backend sees every frame → saves evidence, injects capture nudges |
| Vision | Gemini 3.8 Flash: spine boxes + text + items, JSON schema output | One call per keyframe returns spines + items |
| Decisions | DREX calibrated probabilities (identity, listing match, appraisal) | Calibrated, ~150 ms, text-only |
| Config | pydantic-settings + output contract model | Fail fast; contract enforced |
| Checker | Different Gemini model (Pro) audits identified spines | Protects "≤3% confidently wrong"; same vendor = one key (limit documented) |
| GPU | Modal serverless (local machine: no GPU, 5.6 GB RAM) | torch never installed locally |
| 3D | MapAnything (metric, Apache) or VGGT on Modal → poses + depth; choose by spike | Room area needs multi-view geometry |
| Metric scale | Printed ArUco marker (DICT_4X4_50, 15 cm, ids 0/1). Fallback: standard book-format prior | `cv2.aruco` built-in; book prior = cross-check + fallback (low confidence) |
| Book catalog | Google Books → Open Library fallback, `difflib` fuzzy match | Free, no hardcoded catalogue |
| New prices | Serper.dev `/shopping` with `gl` per locale | Sourced URL per offer; 2,500 free queries |
| Used prices | eBay Browse API `conditions:{USED}`; IN converted from EBAY_US | No Indian used-book API → labelled converted |
| FX | Frankfurter (ECB rates), dated | Free, no key |
| Storage | JSON/JPEG files under `sweeps/<id>/` | No DB needed |
| Report | HTML via f-strings + `html.escape`; floor plan SVG | Brief allows HTML |
| Totals | Python `Decimal`, never LLM | Brief rule |
| Agents | Plain functions per stage, JSON in → JSON out, rerunnable | "Separate, testable stages" |
| Locales | Primary IN/INR, comparison US/USD; table-driven (`LOCALES`) | Locale is a setting |
| Phone | Android Chrome via `cloudflared` HTTPS tunnel | getUserMedia needs HTTPS |

Skipped (YAGNI / unscored): auth, deployment, DB, React, LangGraph/ADK, avatar, sketch gen, policy DB, condition grading, WebXR/ARCore depth (listed under "two more weeks").

## Architecture

```
PHONE (Android Chrome, HTTPS tunnel)
  mic PCM16 16kHz ─┐  camera JPEG 1920px @3fps ─┐   ◄─ agent audio PCM 24kHz, JSON events
                   ▼                            ▼
FASTAPI /ws
  live relay   → Gemini Live: audio + 1fps 768px frames; tools; contextWindowCompression;
                 session resumption (audio+video sessions die at ~2 min without compression)
  frames       → every frame saved sweeps/<id>/frames/NNNNNN.jpg + frames.jsonl (t, blur, glare, motion, key)
  quality      → blur (Laplacian var) / glare (saturated px) / motion (frame diff) → [MONITOR] nudges, rate-limited
  live worker  → keyframes → Flash read → reads/NNNNNN.json → provisional inventory to UI
                 → unreadable spines → "step closer"; art → "ask original or print"
  Modal warm-up fired at sweep start

END SWEEP → pipeline/run.py  (stages rerunnable: python -m pipeline.run sweeps/<id> --from <stage>)
  keyframes → [recon (Modal GPU) ∥ reads (Flash)] → geometry → books → items → pricing → review → packet
```

### Sweep directory contract
```
sweeps/<id>/
  sweep.json        id, captured_at, device, country, currency, t_end, duration_s
  frames/ NNNNNN.jpg   frames.jsonl   notes.jsonl (log_note / exclude_shelf / set_item_fact, timestamped)
  keyframes.json    reads/NNNNNN.json   recon.npz
  geometry.json     books.json   items.json   prices.json   review.json
  crops/ b_001.jpg i_001.jpg
  claim_packet.json  report.html  floorplan.svg  metrics.json
```

### Stage details
- **keyframes**: sharp + no glare + (novelty vs last keyframe or ≥2 s gap), ≤1 per 0.5 s. Same selector live and offline.
- **recon**: ≤150 keyframes resized long side 518 (multiples of 14, no crop) → VGGT → extrinsic, intrinsic, depth, conf (fp16).
- **geometry**: ArUco corners inset 15% → per-corner depth → 3D edge length → scale = marker_m / edge (median over frames, MAD → confidence). No marker → book-format prior. Up = mean camera-down; floor = lowest strong height-histogram peak, SVD refine; ceiling = highest peak ≥2 m above. Wall points (0.3 m–2.0 m) → 5 cm grid → close → largest contour → approxPolyDP; fallback convex hull if gaps. L/W via minAreaRect; floor area = polygon area; wall area gross = perimeter × height (net minus openings = extra field).
- **books**: detections → 3D center/dims (median depth in box) → greedy 3D + text dedup → units (xy gap 0.4 m) → rows (bottom-z gap 0.10 m) → positions → text vote + crop re-read → Google Books/Open Library → `id_confidence = 0.65·title_sim + 0.2·author_sim + 0.15·view_agreement`. ≥0.75 identified; 0.5–0.75 identified + review; <0.5 unidentified (title blank). ISBN/edition only if spine publisher matches. Notes/exclusions applied via timestamp → frame → most central detection.
- **items**: detections (non-book) → 3D extents (p5–p95, PCA) → dedup (same category, <0.6 m) → dims.
- **pricing**: new = median of title-matched local Serper offers (ebook/Kindle/box sets excluded); none → US + FX (`converted: true`); none → blank, excluded. Used = eBay median, converted when foreign. Appraisal (not priced): signed / first edition / leather / gilt / old binding / user note / price > threshold (₹5,000 / $75). (Pub-year rule dropped: catalogue year is the work's, not the copy's — would flag every Dickens paperback.) Items: brand known → priced; else p25–p75 range. Art → needs_appraisal unless user says print. 10 books re-priced for US → `locale_comparison`.
- **review**: rule list (unidentified, low conf, checker mismatch/unsure, appraisal, no price, implausible dims, no scale reference, scale disagreement, room fallback, ceiling missing).
- **packet**: schema from brief + extra fields; totals in `Decimal`; `excluded_from_totals` = count of lines without a price in totals.

## Repo layout
```
app/        server.py  live.py  static/index.html  static/app.js
pipeline/   settings.py metrics.py frames.py vision.py geometry.py books.py items.py pricing.py review.py packet.py run.py marker.py
gpu/        modal_recon.py
eval/       score.py
ground_truth/  books.csv spines.csv prices.csv items.csv room.csv
tests/      test_totals.py test_pricing.py test_geometry.py
docs/       PLAN.md ARCHITECTURE.md FAILURE_LOG.md RESULTS.md REFERENCE_NOTES.md
README.md  .env.example  pyproject.toml
```

## Budgets
- Time to packet < 3.5 min (bar 5). Cost per sweep < $1 (measured in metrics.json).

## Risks → fallback
| Risk | Fallback |
|---|---|
| Reconstruction fails on plain walls | Per-frame metric depth + marker, corner prompts, low confidence → review |
| Spines unreadable at 1080p | Agent asks to step closer per row; crop re-read |
| Dedup over/under-count | Measured by eval → likely the "measured fix" in failure log |
| eBay keys delayed | Used values blank, excluded from totals |
| Live session drops | Session resumption |

## Task checklist

### Phase A — code (no keys needed)
- [x] A1 Project scaffold: pyproject, .env.example, .gitignore, settings
- [x] A2 Marker PDF generator
- [x] A3 frames.py: quality metrics + keyframe selector
- [x] A4 vision.py: Flash frame read, crop re-read, checker
- [x] A5 gpu/modal_recon.py: VGGT on Modal
- [x] A6 geometry.py: scale, floor/ceiling, room polygon, 3D helpers
- [x] A7 books.py: dedup, shelves, dims, resolve, notes/exclusions
- [x] A8 items.py
- [x] A9 pricing.py: Serper, eBay, FX, appraisal, locale comparison
- [x] A10 review.py
- [x] A11 packet.py: totals, JSON, HTML report, floor plan
- [x] A12 run.py orchestrator + metrics
- [x] A13 app/server.py + live.py: relay, tools, nudges, live worker
- [x] A14 static UI (index.html, app.js)
- [x] A15 eval/score.py + ground_truth templates
- [x] A16 tests (totals, pricing, geometry) passing offline
- [x] A17 README, ARCHITECTURE, REFERENCE_NOTES, FAILURE_LOG/RESULTS templates (results filled in Phase D)
- [x] A18 Offline end-to-end test: synthetic room → packet (count 30/30, floor/wall/height within tolerance)

### Phase A2 — revision 2
- [x] A19 Model IDs 3.8 + retries/fallback
- [x] A20 DREX client + identity/listing/appraisal decisions
- [x] A21 pydantic-settings + output contract (tests)
- [x] A22 MapAnything Modal class + metric cross-check
- [x] A23 Catalogue query cleaning (volumes), Books API key reuse

### Phase B — keys + reference (user provides keys)
- [x] B0 Gemini, DREX, HF keys in `.env` (Gemini + DREX verified live)
- [ ] B1 Remaining: Serper, eBay dev, Modal → `.env`; enable Books API; `modal deploy gpu/modal_recon.py`
- [ ] B2 Run reference app, finish REFERENCE_NOTES (kept / changed / threw away)
- [x] B3 Verify current Gemini Live + Flash model IDs (3.8; Flash tested live)

### Phase C — ground truth (user, before tuning)
- [ ] C1 Print markers, measure printed size → MARKER_SIZE_CM
- [ ] C2 Count books, hand-type legible titles → books.csv
- [ ] C3 Hand-measure 20 spines → spines.csv
- [ ] C4 Tape room L/W/H, doors/windows, shelving → room.csv
- [ ] C5 Hand-check 15 prices (same source type) → prices.csv; list ≥8 items → items.csv

### Phase D — test & iterate
- [ ] D1 Geometry spike: test sweep → MapAnything vs VGGT floor area vs tape (go/no-go: within 15%); log both in failure log
- [ ] D2 Full sweeps → eval → fix worst error → measure fix
- [ ] D3 Test on an unseen shelf/room (lab / friend)
- [ ] D4 Fill RESULTS.md, FAILURE_LOG.md (3 worst errors, cost, latency)
- [ ] D5 Fresh-clone setup check (< 15 min)
- [ ] D6 Record one-take demo (≤ 6 min), submit on email thread
