# Ground truth and results

> TODO (Phase C/D): fill from `ground_truth/*.csv` and
> `uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/ --md`.
> Report every shelf, not only the ones that worked.

## Status tracker (2026-10-08)
Book bars are measured on photos A and B (two close-up shelf photos from a public sale post, 84 spines, 57 legible;
answer key in a local eval kit, not committed). Room bars wait for the home test ([HOME_TEST.md](HOME_TEST.md)).

| Measure | Pass bar | Now | Evidence | Status |
|---|---|---|---|---|
| Book count | within 5% | B 40 vs 40; A 46 vs 44 (+4.5%) | photos A, B | pass (photos) |
| Title identification | ≥70% correct, ≤3% confidently wrong | 49/57 = 86% correct, 0 confidently wrong, 3 wrong but flagged | photos A, B | pass (photos) |
| Spine dimensions | within 15% (20 spines) | not measured | home test | open |
| Book prices | within 25% (15 books) | not measured (stability check earlier: 11/12 within 25%) | home test | open |
| Non-book items | ≥80% found, correct category | not measured | home test | open |
| Floor area | within 10% | not measured | home test | open |
| Wall area | within 15% | not measured | home test | open |
| Time to packet | < 5 min (300 s) | books stage 144–1,257 s per photo, 44.5 min on the video replay | free-tier quotas | **fail** |

The 8 missed legible titles on the photos: 5 were read correctly from the spine but the catalogue gave no usable
match (empty answers, at least partly Google Books 429s that were cached as "no such book" until commit 1592e64;
Open Library has no entry for some Indian editions; to re-measure when the quota resets), 2 spines gave no text, 1 resolved to a different Poirot title (flagged for review).

### Time per stage against the 300 s budget
| Stage | Now | Main cause |
|---|---|---|
| Spines (SAM 3.1) | ≈ 8 s | — |
| OCR (Modal CPU) | ≈ 68 s | ≈ 20 s cold start when not warmed |
| Books (catalogue, DREX, crop reader) | 144–2,680 s | Google Books 429 retries (220–287 per photo), Gemini 15 requests/min |
| Checker | 68–708 s | Gemini 15 requests/min; 500/day ran out on the replay |
| Pricing | 86–191 s | Serper ≈ 2 queries per book |

## Ground truth collected
| Item | Value | How |
|---|---|---|
| True book count | | hand count, every shelf |
| Legible spines (hand-read titles) | | `ground_truth/books.csv` |
| Hand-measured spines | 20 | tape/ruler → `spines.csv` |
| Hand-checked prices | 15 | same source type → `prices.csv` |
| Non-book items | | `items.csv` |
| Room (L × W × H) | | tape measure → `room.csv` |

## System vs pass bars (sweep `<id>`, the one in the demo video)
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

## Unseen-room check
| Measure | Own room | Second room |
|---|---|---|
