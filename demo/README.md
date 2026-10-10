# Demo sweep `20261010-131406-5ced` (the sweep in the video)

The tester's bedroom, filmed live on an Android phone with the agent, 154 s, India / INR.

- `20261010-131406-5ced/claim_packet.json` and `report.html`: the packet the phone received, unchanged.
  `frames/` holds the 18 saved frames the packet references (`frame_ref`), and `crops/` holds the spine and item
  crops. The full sweep (462 frames, 3D reconstruction) is kept locally, not committed (155 MB).
- `20261010-131406-5ced-rerun/claim_packet.json`: the same saved sweep rerun from the books stage after the fixes it
  prompted (commits e0d15dd, f7762f1). It gives 12 books instead of 13. Its crop and frame references point to the
  same frames as the live packet.
- Scores against `ground_truth/home-bed/`: [docs/RESULTS.md](../docs/RESULTS.md).

Rescore: `uv run python -m eval.score demo/20261010-131406-5ced/claim_packet.json ground_truth/home-bed/ --md`
