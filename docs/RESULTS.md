# Ground truth and results

Every number below comes from a saved sweep and can be rerun:
`uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/<set>/ --md`.
Runs that missed are reported next to the ones that passed.

## Where each pass bar stands (2026-10-09)

| Measure | Pass bar | Best evidence so far | Status |
|---|---|---|---|
| Book count | within 5% | Real home, current code, 12 runs on 4 sweeps: 5 within 5%. Bed (12 books): 12–18. Shelf + stack (21): 19–20 | **partial**: passes on 3 of 4 sweeps at least once, never on the first bed sweep |
| Title identification | ≥70% correct, ≤3% confidently wrong | Bed 12/12 on every run; shelf + stack 16–17/21 (76–81%); 0 confidently wrong except one duplicate line with a catalogue title (below) | **pass** on real sweeps |
| Spine dimensions | within 15% (20 spines) | no hand-measured spines yet | open |
| Book prices | within 25% (15 books) | no hand-checked prices yet; earlier stability check 11/12 within 25% | open |
| Non-book items | ≥80% found, correct category | no item list yet | open |
| Floor area | within 10% | no tape measurement; the system gave 5.76, 7.86 and 10.5 m² on three sweeps of one room, and nothing on two | open, **likely fail** |
| Wall area | within 15% | as floor area | open, **likely fail** |
| Time to packet | < 300 s | live 418 s and 589 s; with today's speed-ups, about 285 s projected, not yet measured live | **fail until measured** |

The required capture (≥60 books on ≥2 shelving units, ≥8 items, one enclosed room, tape-measured) is the demo sweep
still to be filmed. Its table is at the end.

## Real home sweeps (a tester's room, phone, live agent)

Ground truth: `ground_truth/home-bed/` (12 books laid on a bed, titles hand-read by the tester) and
`ground_truth/home-shelf/` (21 books on a shelf and a flat stack). Each sweep was rerun from the books stage with the
current code (`2657e45`), so the live packets' counts below are from the code of that day.

| Sweep | Truth | Live packet (code of the day) | Current code, full runs | Titles correct | Confidently wrong |
|---|---|---|---|---|---|
| 183215 (bed, 1st) | 12 | 18 | 13, 17, 18 | 12/12 each run | 0, 0, 1 |
| 184524 (bed, 2nd) | 12 | 19 | 13, 13, **12** | 12/12 each run | 0 |
| 194203 (bed, 3rd) | 12 | 18 | **12**, **12**, 13 | 12/12 each run | 0 |
| 203017 (shelf + stack) | 21 | 59 | 19, **20**, **20** | 16, 16, 17 of 21 | 0 |

**Bold** = within 5%. On 12 books, one extra line is already 8.3%.

- The one confidently wrong line (183215, third run) is the second copy of Tom Gates, named by the catalogue as
  'Tom Gates #12 Family, Friends and Furry Creatures [Hardcover] [Jan 01, 2017] LIZ PICHON'. The book is right, but
  the line is a duplicate and the title is not the hand-read one, so the scorer counts it as wrong.
- The count varies between runs on the same frames. Before the reconcile step, the first bed sweep gave 14, 15 or 18
  lines (the crop reader's claims change). The reconcile step then merged 0–5 of them; on two runs Gemini answered
  that all 14 lines were separate books. The causes are in [FAILURE_LOG.md](FAILURE_LOG.md) #1.
- Shelf + stack, run 3: the scorer misses 4 of 21. The City Dog, Boruto and No Onion No Garlic were not named (left
  unidentified or not separated, never guessed). Exam Warriors was named by its Marathi title 'एक्झाम वॉरियर्स',
  which the scorer cannot match to the English name on the tester's list. One extra line, 'Life Is Simple', is not
  on the list; it is below the confidence bar and in the review queue.
- 'Catching Fire' appears at confidence 0.5 (review queue) on the first two bed sweeps. It is not on the tester's
  list; whether that book was in view is still to be confirmed.

### Prices on these sweeps
No hand-checked prices yet, so there is no accuracy figure. Coverage, current code:

| Sweep | Identified books | New price | Used price |
|---|---|---|---|
| 194203 (bed) | 12 | 12 | 4 |
| 200647 | 16 | 14 | 3 |
| 203017 (shelf + stack) | 28 lines | 26 | 6 |

Every price has a URL, seller and retrieval time. Books without a matching listing are left unpriced and excluded
from the totals. The used price exists only where a second-hand seller lists the book on Google Shopping India.

## Close-up shelf photos (title identification, 2026-10-08)

Two close-up shelf photos from a public sale post: 84 spines, 57 legible. The answer key is in a local eval kit
that is not committed, because the photos are not ours.

| Measure | Result |
|---|---|
| Count | B 40 vs 40; A 46 vs 44 (+4.5%) |
| Titles | 49/57 = 86% correct, 0 confidently wrong, 3 wrong but flagged for review |

The 8 missed legible titles: 5 read correctly from the spine but the catalogue gave no usable match, 2 spines gave
no text, 1 resolved to a different Poirot title (flagged).

## Ground truth collected

| Item | Value | Where |
|---|---|---|
| True book count | bed 12; shelf + stack 21; demo room: to do | `ground_truth/home-*/books.csv` |
| Hand-read titles | 12 + 21 | same |
| Hand-measured spines (20) | to do | `ground_truth/spines.csv` |
| Hand-checked prices (15) | to do | `ground_truth/prices.csv` |
| Non-book items | to do | `ground_truth/items.csv` |
| Room (L × W × H, tape) | to do | `ground_truth/room.csv` |

## System vs pass bars: demo sweep `<id>` (the one in the video)

To fill from `eval.score` once the demo room's ground truth is in `ground_truth/`.

| Measure | Pass bar | Result | Pass |
|---|---|---|---|
| Book count | within 5% | | |
| Title identification | ≥70% correct, ≤3% confidently wrong | | |
| Spine dimensions | within 15% (20 spines) | | |
| Book prices | within 25% (15 books) | | |
| Non-book items | ≥80% found, correct category | | |
| Floor area | within 10% | | |
| Wall area | within 15% | | |
| Time to packet | < 5 min | | |
