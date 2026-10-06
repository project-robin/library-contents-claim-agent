# Ground truth and results

> TODO (Phase C/D): fill from `ground_truth/*.csv` and
> `uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/ --md`.
> Report every shelf, not only the ones that worked.

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
