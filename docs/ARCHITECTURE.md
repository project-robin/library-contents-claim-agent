# Architecture

```
 PHONE (Chrome, HTTPS tunnel)                 FASTAPI  app/server.py
 ┌───────────────────────┐   typed replies   ┌──────────────────────────────────────────────┐
 │ camera 1920px @3fps   │ ────────────────► │ relay ──► Gemini Live (text + 1 fps frames)   │
 │ reply box / mic*      │   JPEG frames     │           tools: set_locale, log_note,        │
 │ speaker  ◄────────────┼── PCM16 24k ◄──── │           exclude_shelf, set_item_fact,       │
 │ live inventory ◄──────┼── JSON events     │           end_sweep                           │
 └───────────────────────┘                   │ every frame ─► sweeps/<id>/frames + quality   │
                                             │ quality (blur/glare/motion, in code) ─► [MONITOR] nudge │
                                             │ keyframe ─► Gemini Flash read ─► live inventory │
                                             └──────────────┬───────────────────────────────┘
                                                            │ end_sweep
 pipeline/run.py  (each stage: JSON in -> JSON out, rerunnable with --from)
 keyframes ─┬─► reads (Gemini Flash: items, reader claims) ───────────────┐
            ├─► spines (SAM 3.1, Meta API, one box per spine) ──────────┤
            └─► recon (MapAnything/VGGT on Modal GPU) ─► geometry ─► books ─► checker ─► items ─► pricing ─► packet
                                                          scale       dedup     audit     dims    Serper    totals (Decimal)
                                                          floor/ceil  shelves   (crop)            eBay, FX  review queue
                                                          room shape  dims, ID                              JSON + HTML + SVG

 books, per spine:  SAM box ─► crop ─► PP-OCRv5 words (Modal CPU) ─┬─► claim confirmed / vetoed by its own words
                    Gemini spine with IoU ≥ 0.3 = reader claim ────┘   no claim / vetoed: OCR fragments ─► catalogue ─► DREX
                    still unidentified ─► Gemini crop reader (its OCR words must agree; none readable: capped, review)

 books, per room:   detections ─► 3D clusters ─► merge (related titles ≤ 15 cm) ─► same_book (one catalogue work,
                    one place) ─► frame_cap (unread lines ≤ spines seen in one full view) ─► reconcile (one Gemini
                    call per shelf unit: which lines are one physical book; may only group or drop)

 * mic only with "Voice replies (wear headphones)"; on the loudspeaker the agent heard itself, so replies are typed
```

## How a spine is found and named

Spines come from **SAM 3.1** (Meta API, prompt `book spine`), one image call per keyframe. SAM's video mode tracks at
most 16 objects per frame, fewer than one shelf holds, so the pipeline uses image mode and links spines across frames
itself (3D clustering below). SAM gives one tight box per spine. Gemini's boxes drift onto the neighbouring spine, so a
crop cut from them can show the wrong book: that capped the earlier OCR-on-Gemini-boxes design (FAILURE_LOG #2).

Each spine's identity must come from evidence on **its own crop**:
1. **OCR witness.** PP-OCRv5 reads the crop at 90° and 270° (and 0° when the crop is wider than tall). A detector +
   recognizer cannot invent a title; it can only misspell what is printed. Words repeated across the whole sweep
   (series names, publisher logos) are ignored.
2. **Reader claim.** The Gemini frame read is matched to SAM boxes by IoU ≥ 0.3. Its title is a claim: resolved
   against the catalogue, then confirmed or vetoed by the OCR words (a veto needs 4+ character tokens at ≥ 0.9).
   A claim the OCR words neither confirm nor veto is kept at `CONF_MIN` and sent to review ("title from vision model
   only, no OCR support"). Unmatched Gemini spines are dropped; they have no box to show.
3. **Fragments.** With no claim, or a vetoed one, the OCR words alone go to the catalogue and DREX. The top match
   must beat the runner-up by a clear margin.
4. **Crop reader.** Every spine still unidentified (and not excluded) goes to Gemini as a single crop, 12 per call.
   Its title must agree with the crop's own OCR words, and a disagreement rejects it; when OCR read no usable words it
   is kept at `CONF_MIN` and sent to review.

`id_source` in each book line records which path named it: `spine_read`, `ocr_fragments` or `crop_reader`.

## How a book is counted once

A phone sweep sees each book in 5–25 frames, and on a slanted stack SAM's axis-aligned boxes overlap (76–79% of
boxes on live sweeps vs 4–12% on photos). Detections are clustered in 3D, then four steps remove repeats, each
writing its count to `books.json` stats and its actions to the review queue:
1. **merge**: clusters with related titles ('RAAVAN' / 'RAAVAN: ENEMY OF ARYAVARTA') within 15 cm.
2. **same_book**: lines resolved to one catalogue work within 15 cm and half a spine height (volume numbers must
   agree); unread pieces inside a spine are folded into it; unread boxes under 10 cm are not books.
3. **frame_cap**: a unit's unread lines are capped at the spines Gemini counted in one frame showing all of it.
4. **reconcile**: one Gemini call per shelf unit, with its two fullest frames and every line's crop, OCR and title,
   says which lines are one physical book. Guardrails: it can only group or drop; two different confident titles
   are never merged; a confident line is never dropped; an answer that does not list every line once is ignored.

**Fallbacks.** `SPINE_SOURCE=gemini`, a missing `META_API_KEY` or a SAM failure falls back to Gemini's spine boxes,
with the review note "spines from vision model". Still photos have no geometry: each detection is its own book and
shelf/position come from box order (rows by vertical overlap, top to bottom, then left to right).

**Outside services failing.** A catalogue lookup that times out after its retries returns no candidates, a price
lookup that fails leaves the line unpriced with a review reason, and a failed identity check (e.g. the checker's free
daily quota spent) is verdict `unsure`. Each sends lines to review instead of killing a packet whose books are built.

## Which model does what

| Stage | Model / method | Output |
|---|---|---|
| Conversation, directing the sweep, corrections | Gemini Live (native audio out; typed replies, or voice with headphones) | tool calls → `notes.jsonl` |
| Which lines are one physical book | Gemini (`VISION_MODEL`) on a shelf unit's frames and crops; may only group or drop | `books.json` stats `reconciled` |
| Blur, glare, motion | OpenCV (Laplacian variance, saturated pixels, frame difference) | nudges into the Live session |
| Spine boxes (one per spine) | SAM 3.1 via the Meta API, image mode, prompt `book spine` | `spines/*.json` |
| Printed words on each spine crop | PP-OCRv5 server (PaddleOCR 3.7.0, paddlepaddle 3.2.0), Modal CPU app `library-claim-ocr` | `ocr.json` |
| Spine text claims, item boxes/categories | Gemini 3.8 Flash (3.5 Flash-Lite fallback), schema-constrained JSON, temperature 0 | `reads/*.json` |
| Spines still unidentified after OCR | Gemini Flash on single crops, kept only if the crop's OCR words agree | `id_source: crop_reader` |
| Camera poses + per-pixel depth | MapAnything (metric, Apache) or VGGT-1B on Modal | `recon.npz` |
| Identity | Google Books / Open Library candidates → DREX P(same work) | `id_confidence` (calibrated) |
| Listing match, appraisal | DREX yes/no probabilities | offers kept at P ≥ 0.8; appraisal at P ≥ 0.5 |
| Identity audit | `CHECKER_MODEL` (Gemini) on the crop and the claim | mismatches cleared to blank; failed check = `unsure` |
| Prices | Serper Google Shopping, eBay Browse, Frankfurter FX: **no LLM** | `prices.json` |
| Totals | Python `Decimal` | `claim_packet.json` |

## Where metric scale comes from

VGGT returns geometry in arbitrary units. The scale is fixed by a **printed ArUco marker** of known size, ruler-checked
and set in `.env`:

1. Detect the marker corners in each full-resolution keyframe (OpenCV).
2. Shrink the corner square to 70% so every depth sample lands on the paper, then lift the 4 points to 3D using the
   reconstructed depth.
3. scale = real side ÷ reconstructed side. Take the median over every detection; the spread becomes the confidence.

Cross-check 1: with MapAnything the reconstruction is already metric, so the marker correction should be ~1.0; >10% disagreement lowers confidence and goes to review. Cross-check 2: after scaling, the median upright spine must fall within standard book formats (14–32 cm). If it
doesn't, the room and dimensions go to review. **Fallback**: if no marker is ever seen, scale comes from that
book-format prior (median upright spine = 21 cm). Confidence is then capped at 0.3 and the packet says so.

Using the scaled point cloud:
- **Up and floor.** Up starts as the mean camera-down vector. The floor is the lowest dense horizontal layer,
  refined by a plane fit. The ceiling is the highest dense layer at least 2 m above the floor.
- **Room shape.** Take wall points between 0.3 m and the ceiling. Find the rotation where the walls are sharpest, then
  take the *outermost* wall plane on each side, so a bookcase front never stands in for the wall behind it. An
  occupancy-grid polygon catches non-rectangular rooms (L-shapes): if its area is under 85% of the rectangle, the
  polygon is reported instead.
- **Book dimensions.** The median depth inside the spine box gives a 3D height and thickness, taking the median
  over every view of that spine.
- **Item dimensions.** 10th–90th percentile extents of the 3D points inside the item box.

## Where prices come from

| Figure | Source | Rule |
|---|---|---|
| Book replacement | Google Shopping (Serper, `gl` = country) | title words must match, no other language or script, no sets; then DREX P(listing is a new copy of this book) ≥ 0.8; ebook/Kindle rejected; lower-median offer; URL + retrieval time kept |
| Book replacement, no local offer | US, then GB shopping + Frankfurter FX | `converted: true`, original amount and FX date kept |
| Book used | Second-hand sellers (Bookchor, Kitabay, AbeBooks, eBay…) in the same Google Shopping results; else eBay Browse API (used, fixed price) | DREX P ≥ 0.8 with the seller named; not above the new price; lower-median offer; converted when it is a foreign market |
| Item with legible brand/model | Google Shopping | `priced` |
| Item without brand | Google Shopping, generic query | `range` = interquartile range of the listings |
| Art | none | `needs_appraisal` unless the user says it is a print |
| Nothing found | none | left empty, counted in `excluded_from_totals`, review queue |

Locale is the `LOCALES` table in `pipeline/settings.py` (currency, Google `gl`, eBay market, appraisal threshold).
The packet's `locale_comparison` re-prices 10 books in a second country.
