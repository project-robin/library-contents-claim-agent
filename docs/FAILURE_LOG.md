# Failure log

From live phone sweeps of one home (2026-10-08 to 10, including the demo sweep 5ced; ground truth in [RESULTS.md](RESULTS.md)) and two
close-up shelf photos. Everything tried along the way, with numbers: [DEV_LOG.md](DEV_LOG.md).

## The three worst errors

| # | Error | Root cause | What changed | Before → after (measured) |
|---|---|---|---|---|
| 1 | **Each book counted 2–3 times on live video.** 66, 66 and 59 books for 21 real ones; 18–19 lines for a bed of 12. The photo tests showed none of it. | SAM 3.1 finds the spines, but we keep only its axis-aligned box (its mask format is undocumented). On a slanted spine that box covers the next two: 76–79% of live boxes overlap a neighbour vs 4–12% on photos. A video shows each book in 5–25 frames, read differently each time ('RAAVAN' / 'RAAVAN: ENEMY OF ARYAVARTA'), with half-spine boxes and an orientation that flips. | Rules after 3D clustering: merge related titles within 15 cm; drop unread single-frame boxes; one catalogue work in one place is one book; cap unread lines at the spines seen in one full view. Then one Gemini call per shelf unit that may only group or drop lines (`pipeline/reconcile.py`). The agent now asks for head-on filming. | Shelf + stack: 59 → 31 (rules) → 19–20 (reconcile) → 20, 21 (piece rules, 2026-10-10), truth 21. Bed: 18 / 19 / 18 → 13–18 / 12–13 / 12–13, truth 12. Demo 5ced: live 13 → 12, 12, 12. Correct titles were never lost. **Still open:** run-to-run variance on the first bed sweep (13, 17, 18). |
| 2 | **Titles the spine does not support.** Gemini Flash-Lite, reading whole frames, wrote plausible titles under price stickers; author names ('SIDNEY SHELDON') matched books named after the author; Gemini's boxes put a neighbour's words on a spine. | A vision model completes text it cannot read, and nothing checked its claim against the spine. Two judges (DREX, checker) agreed with a literal author-name match. | Identity must come from words on the spine's own crop: PP-OCRv5 confirms or vetoes each claim, SAM gives the box, and a name with no title is not an identification. Words of neighbouring books' reads do not veto. Unconfirmed claims go to review at 0.5. | Photos: 2 → 0 confidently wrong, 86% of legible titles correct; author-as-title: 18 of 123 IDs → 0. Real sweeps: bed 12/12 titles, 0 confidently wrong on 8 of 9 runs. |
| 3 | **Time to packet 418–589 s** (bar 300 s). | Free-tier quotas. Live frame reads (~50 a minute) queued behind Flash-Lite's 15 a minute (~140 s of backlog at the end); the checker made one call per book (114 s); Serper's 15–45 s tails and 429s (pricing 111 s); the crop reader's catalogue lookups ran one at a time (books 184 s). | One frame read per 3 s; checker 8 books per call; Serper queries in parallel with a 12 s cutoff, 2 tries, 8 in flight; catalogue lookups in parallel and overlapped with OCR. | Books 184 → 30–95 s; checker 114 → 3–30 s; pricing 111 → 53–60 s (empty cache). **Measured live:** 308.8 s (sweep 230330, before the pricing overlap), then 225.5 s (3996) and 243.7 s (demo 5ced). |

Other errors found and fixed: on the first voice sweep (DEV_LOG, sweep 230330) 36 item lines for one bedroom (one bed as 5 lines, the marker card as 'art', sockets) → 17–20, and time to packet 308.8 s with pricing at 145 s (fallback markets and the comparison now overlap: 105 → 77 s on a replay); wrong-edition price matches (a Kannada, Hindi, Czech or Gujarati edition, a different
series volume, a 4-book set) and no used price for any book (DEV_LOG, 2026-10-09); the agent hearing itself on the
phone's loudspeaker and looping on "which currency?" (now typed replies, or voice with headphones).
Found on the demo sweep and fixed: the agent ended one attempt by itself 5 s in (end_sweep now refused before the user
speaks or 30 s); Raavan named by its Hindi edition (near-tie editions go to the closest title); a half-spine read
'THE OAT THE VAYUI' resolved to 'The Vayu Purana' as a 13th book (folded into its neighbour).
**Room:** earlier sweeps gave 5.76, 7.86 and 10.5 m² for one room, or nothing, when the walls were not filmed. The demo
sweep, filmed turning through all four walls, gave 9.14 m² against the builder's 9.29 m² (1.6%); its height, 2.40 m
against the builder's 3.05 m, fails the wall bar. No tape measurement was possible, so which figure is wrong is open.

## Cost per sweep (USD, list prices; most of it ran on free tiers)

| Sweep | Length | Gemini Live | Gemini Flash / Lite | DREX | Serper | GPU recon | OCR | Total |
|---|---|---|---|---|---|---|---|---|
| 203017 (shelf + stack) | 127 s | 2.37 | 0.19 | 0.08 | 0.14 | 0.03 | 0.004 | **2.82** |
| 194203 (bed) | 142 s | 0.50 | 0.09 | 0.07 | 0.10 | 0.03 | 0.002 | **0.79** |
| 200647 | 186 s | 1.79 | 0.13 | 0.07 | 0.11 | 0.06 | 0.003 | **2.16** |
| 5ced (demo, bedroom) | 154 s | 0.84 | 0.15 | 0.05 | 0.11 | 0.04 | 0.004 | **1.19** |

Gemini Live is 63–84% of the cost (70% on the demo): one video frame a second goes into the session. SAM 3.1 has no published price
(0 here).

## Latency per stage (s)

| Stage | Live sweep 200647 (code of 20:06) | Demo 5ced (live) |
|---|---|---|
| spines (SAM 3.1) | 57 | 56 |
| recon (MapAnything, Modal GPU) | 113 | 89 |
| geometry | 14 | 21 |
| OCR (Modal CPU) | 32 | 40 |
| books (incl. reconcile) | 194 | 95 (items, 28, in parallel) |
| checker | 29 | 17 |
| pricing | 53 | 60 |
| **time to packet** | **418** (stages partly overlap) | **243.7** |

## With two more weeks
- A per-spine outline (decode SAM's mask, or fit a rotated box) for the crop and the 3D point: it removes most of
  error 1 at its source instead of after the fact.
- Ask reconcile twice and apply only what both answers agree on, so its result is stable between runs.
- Start recon and SAM on keyframes during the sweep, not after it: they are 170 s of the remaining budget.
- Android WebXR/ARCore depth and poses as a second metric source (no marker needed).
- Condition grading per spine feeding the used value; door and window detection for net wall area.
