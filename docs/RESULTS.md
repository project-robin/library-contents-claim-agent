# Ground truth and results

Every number below comes from a saved sweep and can be rerun:
`uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/<set>/ --md`.
Runs that missed are reported next to the ones that passed.

## Where each pass bar stands (2026-10-10)

| Measure | Pass bar | Best evidence | Status |
|---|---|---|---|
| Book count | within 5% | Demo sweep 5ced (12 books): live packet 13; current code 12, 12, 12. Earlier sweeps, current code: bed 194203 12; shelf + stack 203017 20 and 21 (truth 21) | **pass** on current code; the live demo packet missed by one line |
| Title identification | ≥70% correct, ≤3% confidently wrong | Demo: live 11/12 with 1 wrong edition title; current code 12/12, 0 wrong (3 runs). Shelf + stack 17/21, 0 wrong | **pass** on current code |
| Spine dimensions | within 15% | 12 spines hand-measured: mean error 30–32% | **fail** (thickness too big: axis-aligned boxes) |
| Book prices | within 25% | 12 hand-checked Amazon.in prices: median error 22–23%, 58–73% within 25% | **pass** on the median (the bar does not say median or each) |
| Non-book items | ≥80% found, correct category | 14 items listed by the tester: 11/14 live, 9/14 on a rerun | **fail** (fan and ceiling lights never found) |
| Floor area | within 10% | 9.14 vs 9.29 m² (1.6%) | **pass**, against the builder's 10 × 10 ft (not tape) |
| Wall area | within 15% | 29.1 vs 37.2 m² (21.6%) | **fail** against the builder's 10 ft height: the system measured 2.40 m |
| Time to packet | < 300 s | live 225.5 s (sweep 3996) and 243.7 s (demo 5ced) | **pass** |

What differs from the brief's required capture, and why: the tester's home has 12 books in the bedroom and 21 in all
(not ≥60 on ≥2 units); the room dimensions are the builder's 10 × 10 × 10 ft, because no tape measure was available; 12 spines were measured and 12 prices checked by hand (not 20 and 15).

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
| True book count | bedroom 12; shelf + stack 21 | `ground_truth/home-*/books.csv` |
| Hand-read titles | 12 + 21 | same |
| Spines measured by the tester with a ruler | 12 (height and thickness) | `ground_truth/home-bed/spines.csv` |
| Prices checked by the tester on Amazon.in | 12 (2026-10-10) | `ground_truth/home-bed/prices.csv` |
| Non-book items listed by the tester | 14 | `ground_truth/home-bed/items.csv` |
| Room L × W × H | 3.048 × 3.048 × 3.048 m, the builder's figure, **not tape-measured** | `ground_truth/home-bed/room.csv` |

## System vs pass bars: demo sweep `20261010-131406-5ced` (the one in the video)

The tester's bedroom, live agent, 154 s sweep, India / INR. Ground truth `ground_truth/home-bed/`.
"Live packet" is what the phone received. "Current code" is the same saved sweep rerun from the books stage after
the fixes it prompted (commits e0d15dd, f7762f1); geometry and recon are unchanged.

| Measure | Pass bar | Live packet | Pass | Current code (rerun) | Pass |
|---|---|---|---|---|---|
| Book count | within 5% | 13 vs 12 (8.3%) | ❌ | 12 vs 12, three runs | ✅ |
| Title identification | ≥70% correct, ≤3% confidently wrong | 11/12; 1/11 wrong (Raavan under its Hindi edition title) | ❌ | 12/12; 0 wrong | ✅ |
| Spine dimensions | within 15% (12 spines) | mean 32.0%; 45% within 15% | ❌ | mean 29.7%; 50% within 15% | ❌ |
| Book prices | within 25% (12 books) | median 21.7%; 73% within 25%; 11/12 priced | ✅ | median 21.8%; 70% within 25%; 10/12 priced | ✅ |
| Non-book items | ≥80% found, correct category | 11/14 (79%) | ❌ | 9/14 | ❌ |
| Floor area | within 10% | 9.14 vs 9.29 m² (1.6%) | ✅ | same | ✅ |
| Wall area | within 15% | 29.12 vs 37.16 m² (21.6%) | ❌ | same | ❌ |
| Time to packet | < 5 min | 243.7 s | ✅ | (live only) | |

- Room: 3.24 × 2.82 m measured vs the builder's 3.05 × 3.05 m (6–8%); height 2.40 m vs 3.05 m. Floor area agrees,
  so the scale is right; whether the builder's 10 ft is the ceiling height or floor to floor could not be checked.
- Items: matched by category count, as in `eval/score.py`. Found: bed, bedsheet, wardrobe, pillows, bolster, temple,
  clock, AC, marble-top table, dressing table. Never found: ceiling fan, ceiling lights, folded hoodie, baby photo.
  The packet also lists items not on the tester's list (Spider-Man art on the headboard, which the tester confirmed
  as a print during the sweep; curtain; calendar), and three lines for one bedsheet.
- Demo video (https://drive.google.com/file/d/13-gCkYkY3VdGch1Pxio01V_SUFxTr0C6/view?usp=sharing): one unedited phone screen recording, 8 min 12 s (the brief asks for 6 min or less: the sweep took 2 min 34 s and the packet 4 min 4 s), from before Start to the saved PDF. Sweep starts
  0:06, the user ends it at 2:40, the packet arrives at 6:44 (243.7 s later) and the agent reads its summary; the
  report is opened at about 7:10. The PDF the tester saved shows the live packet (13 books).
- Cost of this sweep: $1.19 at list prices (Gemini Live $0.84).
- An earlier attempt the same day (`20261010-131310-d3d6`) ended after 5 s: the agent called end_sweep by itself.
  The server now refuses that before the user has spoken or before 30 s (commit 2b38fa3).
