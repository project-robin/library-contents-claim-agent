# Development log

Everything tried, measured and changed, in order. The one-page summary is [FAILURE_LOG.md](FAILURE_LOG.md).

Title identification is measured on two hand-labelled close-up shelf photos (79 spines, answer key per spine:
legible full / partial / none). Pass bar from the brief: ≥70% of legible spines correct, ≤3% confidently wrong.

## Found during development (synthetic tests)
- **Wall peaks clipped.** The histogram range was cut at the 0.5–99.5 percentiles. Those percentiles fall *inside*
  the outermost walls, so the room collapsed to a point on the L-shaped test room. Fix: pad the range. Covered by
  `tests/test_geometry.py`.
- **Series volumes merged.** "Volume 1" and "Volume 2" score 0.875 string similarity, so adjacent volumes merged
  into one book (22 of 30 counted). Fix: titles whose numbers differ are capped at 0.5 similarity. After the fix:
  30 of 30 (`tests/test_pipeline_offline.py`).

## The three worst errors
| # | Error | Root cause | Fix | Before → after |
|---|---|---|---|---|
| 1 | **Author names identified as titles.** Stacked books often show only "SIDNEY SHELDON"; it matched catalogue works literally named after the author (DREX P=0.99, checker said "match"). | A name was searched as a title, and both judges agreed with a literal match. | `books.author_only`: a name with no title is not an identification. | 18 of 123 IDs wrong on a replayed shelf → 0 |
| 2 | **Invented titles under price stickers.** Gemini Flash-Lite reading whole frames writes a plausible title for a spine it cannot see. | A VLM completes text it cannot read; nothing checked its claim against the spine. | Path A: a PP-OCRv5 witness on each crop vetoes claims its own words contradict. Plan B: identity must come from words on the spine's own crop; Gemini is only a claim to confirm. | Flash-Lite: 10/40 invented on photo B. Path A witness: 7 → 6 wrong. Plan B (photos A+B, 57 legible spines, real pipeline): 2 → 0 confidently wrong; the 2 claims OCR cannot back ('Evil Under the Sun' under a sticker, a translated 'The Hollow') are kept at `CONF_MIN` and sent to review |
| 3 | **Boxes on the wrong spine.** Crops and frame references showed the neighbouring book. | Gemini's spine boxes drift sideways on dense shelves, so the OCR witness read the wrong spine: this is why Path A only moved 7 → 6. | Plan B: spines from SAM 3.1 (one tight box per spine); Gemini claims matched to SAM boxes by IoU ≥ 0.3. | SAM test: photo B 40/40 spines in 4.6 s; photo A 46 boxes (incl. a flat book, leaning books) in 3.0 s. End-to-end on the two photos: count 86 vs 84; correct legible 42/57 → 49/57 (86%) with Flash-Lite claims once every unread crop goes to the crop reader; OCR + crop reader alone 35/57 (61%), 0 confidently wrong |

Gemini 3.8 Flash never invented a title, but on dense photos it timed out at 5 minutes via the API, twice, so it
should only see single crops, not whole dense frames.

Also fixed: **price stability**, books priced within 25%: 9/12 → 11/12.

## OCR engine bake-off
Both photos, rotated clockwise and counter-clockwise (spines are vertical), CPU on Modal
(local eval kit, not committed: the photos come from a public sale post). Title-word recall is as recorded in `gpu/modal_ocr.py`; the other columns are raw outputs of the bake-off JSON.

| Engine | Load s | s per rotated photo | Words returned (A cw/ccw, B cw/ccw) | Result |
|---|---|---|---|---|
| **PP-OCRv5 server** (PaddleOCR) | 30.1 | 9.5–14.1 | 83 / 89, 148 / 147 | best title-word recall: 86% / 49% → chosen |
| docTR + PARSeq | 18.2 | 7.7–11.2 | 213 / 204, 141 / 161 | behind PP-OCRv5 |
| OpenOCR SVTRv2 (mobile / server) | 14.7 / 75.6 | 1.4–1.5 / 15.4–17.0 | 48–112 / 77–113 | behind PP-OCRv5 |
| EasyOCR | 11.3 | 5.3–6.8 | 20 / 51, 29 / 60 | behind PP-OCRv5 |
| RapidOCR | 0.9 | 1.0–1.4 | 53 / 59, 50 / 60 | behind PP-OCRv5 |
| OvisOCR2 (VLM, L4 GPU) | — | 1–55 | — | loops ("1\n\n1…", "LEED LEED…") to the 4,096-token cap |
| TeleOCR (VLM, L4 GPU) | 20.2 | 0.5–6 | — | returned "NULL" on every rotated photo |

Gap: per-engine recall for the five engines behind PP-OCRv5 is not recorded in the bake-off outputs.

On SAM crops, OCR reads honest fragments ("TO ROW" for *Gone Tomorrow*, "HERCULE POIRO", "CURTAIN: POIROT'S LAST
CASE"); about 3 of 40 crops on photo B read nothing (script font + sticker) and go to the crop reader.

## Video replay, one bookcase (Plan B, 2026-10-08)
A 720p bookshelf-tour video replayed as a sweep (15 keyframes, `--from spines --skip recon`). The packet builds:
**271 books, down from 355** with Gemini spine boxes, 141 identified (86 `spine_read`, 51 `crop_reader`,
4 `ocr_fragments`). No answer key exists for this shelf, so correctness is measured only on the photos above.

It needed three robustness fixes on the way, each found by this run: a Google Books timeout after retries, the
checker's free 500/day quota running out mid-sweep, and Serper shopping 504s. All three used to kill a packet
whose books were already built; now they send lines to review instead (see ARCHITECTURE, "Outside services failing").
Because of the spent quota, all 141 checker verdicts on this run are `unsure` (checker unavailable).

### Cost per sweep
| Component | USD | Source |
|---|---|---|
| SAM 3.1 (15 keyframe images) | 0 (placeholder) | no list price published; `SAM_COST_PER_IMAGE` |
| PP-OCRv5 on Modal CPU | ≈ 0.007 | ≈ 68 s × `OCR_COST_PER_S` (unverified list price) |
| Gemini reads + crop reader + checker | 0 billed (free tier) | token counts for the books stage lost when the run crashed |
| Reconstruction GPU | 0 | recon skipped on this replay (cached) |
| DREX (15 calls) + Serper (32 uncached queries) | 0.047 | `metrics.json`, pricing stage; most queries came from the price cache |

### Latency per stage
| Stage | s | Source |
|---|---|---|
| spines (SAM 3.1, 15 images, 8 in parallel) | ≈ 8 | file timestamps |
| ocr (Modal CPU, incl. cold start) | ≈ 68 | file timestamps |
| **spines + ocr** | **≈ 76** (target ≤ 90) | |
| books (OCR, catalogue, DREX, crop reader) | ≈ 2,680 (44.5 min) | file timestamps; almost all catalogue 429 back-off and Gemini timeouts on the free tier |
| checker | not measured | free daily quota spent |
| pricing | 86 | `metrics.json` |
| time to packet | not measured end to end | the run was resumed per stage after each fix |

The books stage is the latency problem: on free-tier quotas, catalogue rate limits and crop-reader timeouts dominate.
Next step would be a time budget for catalogue lookups (spines past it go to review unidentified).

## Live phone sweeps, one home (21 books, 2026-10-08)

Three live sweeps by a tester gave 66, 66 and 59 books for 21 real ones. The offline tests had shown none of this,
because they were straight-on still photos: one frame, upright spines, separate boxes. The live sweeps were video
(69 to 106 keyframes) with a flat stack filmed from above and from the side.

| | Photos A, B | Live sweeps |
|---|---|---|
| SAM boxes overlapping a neighbour's box (IoU > 0.2) | 4%, 12% | 76 to 79% |
| Frames per book | 1 | 5 to 25 |

SAM 3.1 finds the spines. The trouble comes from what we keep: only its axis-aligned box. Its mask uses an
undocumented encoding (`one_bit`/`lossless`; base-85-like text that does not decode as plain base-85 or deflate).
On a slanted spine that box covers the next two spines, which caused three failures:
1. **Duplicates.** One book read differently across frames ('RAAVAN' / 'RAAVAN: ENEMY OF ARYAVARTA'), or with its
   orientation flipping (a slanted box is near square), or its author line read as a title, became 2 or 3 books.
   Fix: `books.merge()` joins related titles within 15 cm; reads of under 4 letters ('4', 'THE') are not titles.
   `books.seen_once()` drops boxes seen in a single frame and never read, in sweeps of 20+ frames. A rule that merged
   by position alone (1.2 cm) joined different books, because overlapping boxes share one centre. It was dropped.
2. **Neighbour text vetoed right titles.** The crop of The Secret of the Nagas read 'RUSKIN BOND ... ROOF', and the
   OCR veto cleared the correct claim. Fix: `spine_ocr.witness(neighbours=)` ignores words of other books' reads in
   the same sweep. A claim that drifted onto the neighbour now loses OCR support and goes to review at 0.5.
3. **Filming.** The live agent now asks for the phone to face the spines head-on, crouching for a flat stack.

Offline rerun of the books stage (no Gemini quota left that day, so no crop reader and no checker in either column):

| Sweep | Count before → after (truth 21) | Distinct right titles | Confidently wrong |
|---|---|---|---|
| 182930 | 66 → 39 | 14 → 19 | 1 → 0 |
| 184815 | 66 → 43 | 16 → 14 (before had the crop reader) | 2 → 1 |
| 203017 | 59 → 31 | 13 → 17 | 2 → 1 |
| Photo A / B (truth 44 / 40) | 46 / 40 → 46 / 40 | 27/37, 18/20 → 28/37, 19/20 | 0 → 0 |

Speed (time to packet was 589 s on sweep 203017; target 300 s):

| Stage | Before | After | Change |
|---|---|---|---|
| Waiting for live frame reads | ≈ 140 s | expected ≈ 0, not yet measured live | one Gemini read per 3 s of keyframes (`READ_GAP_S`), not ~50 a minute against Flash-Lite's 15 a minute; titles on 203017: 17/21 with every 3rd read vs 15/21 with all |
| checker | 114 s | 30-33 s | 8 spines per call (`review.CHECK_BATCH`); on 62 photo IDs, hallucinations caught one per call 3/5, batched 2/5 and 5/5 on two runs |
| pricing (empty cache) | 108-134 s | 50-89 s | a book's 3 Serper queries in parallel, items alongside books, a 12 s cutoff and retry on Serper's 15-45 s tails, at most 8 requests in flight (48 drew 111 HTTP 429s) |

A full rerun from SAM onward, with cold GPU and OCR containers (a live sweep warms them while filming), took 345 s:
recon 98, OCR 65, books 133 (including OCR), checker 30, pricing 69 (with the 429s, before the cap). These test runs
spent the Serper key's credits, so the capped pricing is measured once (80 s, no 429s, 13 timeouts).

Count is still +48 to +105% on the live sweeps: unread boxes and wrong claims on the overlapping stack. The fix that
remains is a per-spine outline (a decoded SAM mask, or a rotated box) for the crop and the 3D point.

### Voice intake loop on the phone (2026-10-09)

Testers on Chrome for Android: the agent kept asking "which country and currency?". Echo cancellation was off (it
forces Android's quiet call mode) and the mic was gated while the agent's audio played, plus 0.3 s.
A fake phone (`echo_loop`: plays the agent's audio back into the mic after a set delay, through the same gate) could
not reproduce it: no loop at 0, 0.5, 1 or 2 s output delay, at full echo volume, or with the gate removed. A spoken
"India. Indian rupees" (espeak) was heard and `set_locale` was called, so the harness reached the agent. Untested
candidates: the phone's own audio path (speaker distortion, room), and a reply clipped by the gate never transcribed,
so `set_locale` stayed refused (the agent does call it before the reply; the server refuses it, as designed).
Fix: the phone sends no audio. The agent still speaks; replies are typed and reach Live as `[USER]` turns, which
unlock `set_locale` at once. 3/3 harness runs: one locale question, `set_locale` accepted, intake moves to the marker.
The server log now records each agent turn, typed reply and tool call.

### Reconcile check: one vision call per shelf unit (2026-10-09)

After the 3D merge rules, the bed of 12 books still came out as 14-19 lines: unread slices of spines and second
views of a book on a slanted stack, whose boxes overlap and share one centre. `pipeline/reconcile.py` sends the
unit's two fullest frames plus every line's crop, OCR text and claimed title to Gemini and asks which lines are one
physical book. It may only group or drop existing lines; two confident titles are never merged and a confident line
is never dropped; an answer that does not list every line once is ignored; each action goes to the review queue.

Same input, before -> after (books stage rerun, two runs each):

| Sweep | Truth | Run 1 | Run 2 | Correct titles |
|---|---|---|---|---|
| 194203 (bed) | 12 | 14 -> 12 | 14 -> 12 | 12/12, 12/12 |
| 183215 (bed) | 12 | 18 -> 13 | 18 -> 17 | 12/12, 12/12 |
| 184524 (bed) | 12 | 19 -> 13 | 19 -> 13 | 12/12, 12/12 |
| 203017 (shelf + stack) | 21 | 22 -> 19 | 22 -> 20 | 16, 16 |

No correct title lost, no confident wrong title added. The answer varies between runs (183215: 13 vs 17). On a
synthetic room of blank rendered spines, the model first dropped 20 of 30 as "not a book"; the prompt now says a
covered or unreadable spine is still a book, since the brief's covered test book must be counted.


### Used value for every market (2026-10-09)

Every book's used value was blank: India has no used-book API and no eBay keys were set, so the brief's second
figure per book was missing on every live packet. The three Google Shopping queries per book already returned
second-hand sellers (Bookchor, Kitabay, AbeBooks, Biblio, eBay), which were filtered out of the new price. They are
now the used pool: no extra query, DREX told the seller, and a used offer above the new price dropped (a collector
or import listing is not what an ordinary used copy fetches). eBay stays the fallback when keys are set.

Checking those prices by hand showed DREX passing wrong listings, on new prices as well: 'Flyte By Mark Zug' as The
Oath of the Vayuputras, 'Think Like a Monk (Kannada)', 'Ikigai (इकिगाई)', a Czech Will Grayson, a Gujarati Raavan,
'Tom Gates #24: Pesky Pets And Parties' for Family, Friends and Furry Creatures, a Spanish 'Libro …' and a 4-book
set. `pricing.same_book_title` now runs before DREX: most title words present (or the main title with only the
author left over), no marked language or other script, no Libro/Livre, no set the title does not name.

Saved sweeps rerun from the price cache (Serper $0):

| Sweep | Identified | New price | Used price | Bed books' new total |
|---|---|---|---|---|
| 194203 (bed) | 12 | 12 | 4 (was 0) | ₹2,710 (was ₹3,118 with 3 wrong-edition matches) |
| 200647 | 16 | 14 | 3 (was 0) | |
| 203017 (shelf + stack) | 28 | 26 | 6 (was 0) | |

Used coverage is 25–33% of identified books: most Indian titles have no second-hand listing on Google Shopping. Those
lines say "no retrievable used price" and are left out of the used total, as the brief asks.

### Voice replies with headphones (2026-10-09)

Typed replies fixed the loudspeaker loop but dropped the brief's "voice in". A "Voice replies (wear headphones)"
switch before Start streams the mic again (echo cancellation on, half-duplex gate kept); typing always works.
`sweep.json` records `voice_replies` and the server logs what the user said.

## Live sweep 230330 (bed, voice replies, 2026-10-09)

First sweep with headphones and voice: intake in 13 s (one question, `set_locale` on "Yes"), art answers recorded by
voice. Packet: 13 lines for 12 books (12/12 titles; one extra line 'Hē kyria me to gati' at 0.83, confidently
wrong); room 3.10 × 2.86 × 2.48 m, floor 8.87 m² (marker spread 0.5%, metric model agrees within 4.9%; the tester's
rough "10 × 10 ft" is 9.29 m²). Time to packet **308.8 s**: spines 61, recon 87, geometry 15, OCR 31, books 56,
checker 3, pricing **145**.

Four fixes from that run:
1. **Pricing 145 s.** A book with no Indian listing searched India, then US, then GB on Serper's slow tails (108 s),
   and the 10-book comparison waited for it (38 s). The fallback markets now run in parallel and each comparison
   starts when its book is priced. Replay of this sweep's pricing from its cache (2 s queries, 24 s tails on the
   slow book): 105 s → 77 s, same 11 books priced and 10 compared.
2. **36 item lines for one bedroom.** One bed as 5 lines in 4 categories, the wardrobe 3 times, sockets and switches,
   and the printed marker card twice as 'art' (the phantom "abstract art" the agent asked about). Now: item boxes
   showing the ArUco marker are dropped (OpenCV, 12 detections here), and one Gemini call over the item crops may
   only merge lines or drop marker cards and building fixtures. Code keeps a line the user answered about, keeps
   the larger kind of object when merging, and never folds bedding, rugs, decor or art into furniture (one run of
   three had folded bed, bedsheet and pillows into the bedsheet line and lost the bed). An id the answer skips or
   repeats is kept as it is (on 36 lines one answer did, and the old all-or-nothing rule threw it away).
   Three runs: 36 → 20, 17, 19 lines; the bed kept as one line every time; items range ₹2.0–5.7 lakh → ₹1.0–3.5 lakh
   on the first measured run. The items stage now runs alongside books and the checker, so its 10–15 s vision call
   does not add to time to packet.
3. **"There is no photograph on the wall."** The agent could only apologise. `remove_item` (by id, or by
   description when the agent has none) records the removal; the items stage drops the line and the review queue
   says so.
4. **The mic stayed on after the sweep** and the agent answered the room's background talk. The phone stops the mic
   at End sweep; typing still works.

## 2026-10-10: demo sweeps in the tester's bedroom

The first demo attempt (`003957-3996`) was filmed in another part of the home (the shelf books and other furniture),
so it could not be scored against the bedroom's list; it did give the first live time to packet under 300 s (225.5 s).
The tester then repeated the bedroom.

1. **The agent ended a sweep by itself 5 s in** (`131310-d3d6`): it greeted, then called end_sweep before the user
   said anything. The server refuses end_sweep before the user has spoken or before 30 s; the End button is
   unchanged. It also asked about a portrait after the sweep had ended (a frame read still in flight); art questions
   now stop at end_sweep.
2. **Demo sweep `131406-5ced`:** time to packet 243.7 s; floor 9.14 vs 9.29 m² (builder's figure); 13 lines for 12
   books, one of them Raavan under its Hindi edition title.
   - Editions of one work tie on DREX's P(same work) (0.9805 Hindi vs 0.958 English). Among candidates within 0.05
     of the best, the title closest to the spine read now wins.
   - A half-spine read 'THE OAT THE VAYUI' resolved to 'The Vayu Purana' next to The Oath of the Vayuputras. A read
     that fits a neighbour's title (≥ 0.6, and 0.2 better than its own) is folded into it.
   - Reruns then gave 12, 13, 14 lines: a second box 1.3 cm along the flat Othello spine (the piece test measured
     flat spines on the wrong axes), the crop reader's 'William Shakespeare's Othello', and garbled reads no
     catalogue matched ('TOM GATES FAMI FURR', 'WILLIAM SHAKESPEARE'). The piece rules now cover unidentified lines
     with a read, flat pieces along the spine, and a crop-reader title that starts or ends a neighbour's.
   - The first version of these rules dropped the shelf sweep from 20 to 17 lines: Exam Warriors, named in Marathi,
     had folded into The Ship of Secrets (a Devanagari title scores 0 against any read), and two upright boxes into
     flat books. Fixed by requiring a Latin title, an absolute match, and the same orientation.
   - Reruns from books: demo 12, 12, 12 (truth 12); bed 194203 12; shelf 203017 20, 21 (truth 21). Titles
     unchanged, 0 confidently wrong.
3. **The agent spoke about things already out of view.** In the demo video it asked to "step closer to this row"
   after the user had moved on, and said "you've moved away from the books" during the wall turn. Gemini Live itself
   was fast (a spoken "print" became a tool call in 0.8 s). The lag came from the side reader: every 3 s a keyframe
   goes to a Gemini Flash read of spines and items, and its answers came back 18.6 s after the frame on median (p90
   59 s, max 67 s); unreadable-spine warnings and art questions were forwarded from those reads. Now:
   - a monitor message from a frame more than 4 s old is dropped and counted (`monitor_stale_dropped` in sweep.json);
     the reads still feed the inventory and the packet;
   - the agent coaches from the video it sees (spines small, blurry or slanted) and talks only about the current view;
   - the wall turn, floor and ceiling corner are in its instructions, so it does not send the user back to the books;
   - it asks "original or print?" about art it sees, and `set_item_fact` takes a description; the items stage gives
     the answer to the art line sharing a distinctive word. A later frame read of the same piece does not ask again.
   Checked on a real Live session: told about a framed Spider-Man print, it called set_item_fact with a description.
