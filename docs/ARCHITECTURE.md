# Architecture

```
 PHONE (Chrome, HTTPS tunnel)                 FASTAPI  app/server.py
 ┌───────────────────────┐   mic PCM16 16k   ┌──────────────────────────────────────────────┐
 │ camera 1920px @3fps   │ ────────────────► │ relay ──► Gemini Live (audio + 1 fps frames)  │
 │ mic                   │   JPEG frames     │           tools: set_locale, log_note,        │
 │ speaker  ◄────────────┼── PCM16 24k ◄──── │           exclude_shelf, set_item_fact,       │
 │ live inventory ◄──────┼── JSON events     │           end_sweep                           │
 └───────────────────────┘                   │ every frame ─► sweeps/<id>/frames + quality   │
                                             │ quality (blur/glare/motion, in code) ─► [MONITOR] nudge │
                                             │ keyframe ─► Gemini Flash read ─► live inventory │
                                             └──────────────┬───────────────────────────────┘
                                                            │ end_sweep
 pipeline/run.py  (each stage: JSON in -> JSON out, rerunnable with --from)
 keyframes ─► reads (Flash) ─┐
          └─► recon (VGGT on Modal GPU) ─► geometry ─► books ─► checker ─► items ─► pricing ─► packet
                                            scale       dedup     audit     dims    Serper    totals (Decimal)
                                            floor/ceil  shelves   (Pro)             eBay, FX  review queue
                                            room shape  dims, ID                              JSON + HTML + SVG
```

## Which model does what

| Stage | Model / method | Output |
|---|---|---|
| Conversation, directing the sweep, voice corrections | Gemini Live (native audio) | tool calls → `notes.jsonl` |
| Blur, glare, motion | OpenCV (Laplacian variance, saturated pixels, frame difference) | nudges into the Live session |
| Spine boxes, spine text, item boxes/categories | Gemini Flash, schema-constrained JSON, temperature 0 | `reads/*.json` |
| Close re-read of uncertain spines | Gemini Flash on crops | votes |
| Camera poses + per-pixel depth | VGGT-1B (pretrained) on Modal | `recon.npz` |
| Identity | Google Books / Open Library + fuzzy match (difflib) | `id_confidence` |
| Identity audit | Gemini Pro on the crop and the claim | mismatches cleared to blank |
| Prices | Serper Google Shopping, eBay Browse, Frankfurter FX: **no LLM** | `prices.json` |
| Totals | Python `Decimal` | `claim_packet.json` |

## Where metric scale comes from

VGGT returns geometry in arbitrary units. The scale is fixed by a **printed ArUco marker** of known size, ruler-checked
and set in `.env`:

1. Detect the marker corners in each full-resolution keyframe (OpenCV).
2. Shrink the corner square to 70% so every depth sample lands on the paper, then lift the 4 points to 3D using the
   reconstructed depth.
3. scale = real side ÷ reconstructed side. Take the median over every detection; the spread becomes the confidence.

Cross-check: after scaling, the median upright spine must fall within standard book formats (14–32 cm). If it
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
| Book replacement | Google Shopping (Serper, `gl` = country) | title words must all match; ebook/Kindle/sets rejected; lower-median offer; URL + retrieval time kept |
| Book replacement, no local offer | US, then GB shopping + Frankfurter FX | `converted: true`, original amount and FX date kept |
| Book used | eBay Browse API, USED + fixed price, books category | lower-median matched listing; converted when it is a foreign market |
| Item with legible brand/model | Google Shopping | `priced` |
| Item without brand | Google Shopping, generic query | `range` = interquartile range of the listings |
| Art | none | `needs_appraisal` unless the user says it is a print |
| Nothing found | none | left empty, counted in `excluded_from_totals`, review queue |

Locale is the `LOCALES` table in `pipeline/settings.py` (currency, Google `gl`, eBay market, appraisal threshold).
The packet's `locale_comparison` re-prices 10 books in a second country.
