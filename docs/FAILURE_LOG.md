# Failure log

> TODO (Phase D): the three worst errors from the real run, each with its root cause, the fix, and the measured
> before/after numbers (rerun with `python -m pipeline.run sweeps/<id> --from <stage>`).

## Found during development (synthetic tests)
- **Wall peaks clipped.** The histogram range was cut at the 0.5–99.5 percentiles. Those percentiles fall *inside*
  the outermost walls, so the room collapsed to a point on the L-shaped test room. Fix: pad the range. Covered by
  `tests/test_geometry.py`.
- **Series volumes merged.** "Volume 1" and "Volume 2" score 0.875 string similarity, so adjacent volumes merged
  into one book (22 of 30 counted). Fix: titles whose numbers differ are capped at 0.5 similarity. After the fix:
  30 of 30 (`tests/test_pipeline_offline.py`).

## Worst errors in the real run
| # | Error | Root cause | Fix | Before → after |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Cost per sweep (from `metrics.json`)
| Component | USD |
|---|---|

## Latency per stage (from `metrics.json`)
| Stage | s |
|---|---|

## With two more weeks
- Android WebXR/ARCore depth and poses as a second metric source (no marker needed).
- Condition grading per spine feeding the used value.
- Door and window detection to report net wall area.
- A learned spine segmenter (pretrained, e.g. SAM 2 prompted by the detector boxes) for tighter boxes on thin spines.
