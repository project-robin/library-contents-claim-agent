# Home test: 20 books and one room

This test measures six pass bars in a room without 60 books. It takes about 1.5 hours. Most of that time is the
hand measurements, which you do before the sweep.

Stated gap: the brief asks for at least 60 books on at least 2 shelving units. Book count and titles are also
measured on photos A and B (84 spines). Say this in `docs/RESULTS.md`.

## 1. Prepare the room (15 min)

1. Put the 20 books on **two** surfaces, for example a shelf and a table: 10 and 10. This copies "2 shelving units".
2. Put most books upright with the spine to the front. Lay 2 books flat in a stack. Put 1 book with its spine hidden
   or covered (a sticker or paper). The system must log that one as unidentified, not guess it.
3. Make sure the room has at least 8 non-book items (lamp, chair, table, rug, framed picture, electronics, plant).
4. Print `assets/aruco_markers_A4.pdf` at 100% (actual size). Measure the black square with a ruler. Write the value
   in `.env` as `MARKER_SIZE_CM`.

## 2. Collect ground truth before the sweep (45 min)

Fill these files in `ground_truth/`. Do not look at system output first.

| File | What to write | How |
|---|---|---|
| `books.csv` | One row per book, all 20: shelf, position, title, author, `legible` = y/n | Read each spine from where the camera will be. `y` only if you can read the title from there. |
| `spines.csv` | All 20 books: title, height_cm, thickness_cm | Ruler. Height = top to bottom of the spine. Thickness = spine width. |
| `items.csv` | One row per non-book item: category, description, brand_model | Use the packet's categories exactly: shelving, furniture, lighting, art, rug, electronics, appliance, decor, plant, textile, other. |
| `room.csv` | length_m, width_m, height_m, floor_area_m2, wall_area_m2, shelved_wall_area_m2 | Tape measure. floor = L × W. wall = 2 × (L + W) × H. shelved = width × height of each shelf front. |
| `prices.csv` | 15 of the books: title, amount, currency, source, url, retrieved_at | See step 3. |

## 3. Hand-check 15 prices (20 min)

The system's replacement cost is the lower-median **new** offer on Google Shopping in your country. Use the same
source type:

1. Open Google Shopping (shopping.google.com) set to India.
2. Search the title and author.
3. Keep only offers for a **new** copy of **this** book (not a different volume, not used, not a bundle).
4. Sort those prices from low to high. Write the middle price; with an even count, write the lower of the two
   middle prices.
5. Write the URL of that offer and today's date.

## 4. Check quotas, then sweep (10 min)

Free tiers run out per day. A sweep on a spent quota gives blank titles. Run this first:

```bash
uv run python -m eval.preflight
```

Start only when every line says `ok`. Then start the app (README, "Run a sweep") and open it on the phone.

### How the live sweep works

1. You tap **Start sweep**. The Gemini Live agent greets you and asks the country and currency. You type the answer in the box under the buttons.
2. The agent asks you to lean the marker against a shelf. Then you walk the room once, slowly.
3. The phone sends audio and camera frames to the server. The server keeps sharp keyframes and checks blur, glare and
   motion. When a frame is bad, the agent tells you ("slow down", "step closer").
4. Each keyframe goes to Gemini at once. The inventory list on screen fills in while you walk.
5. You can correct the agent by typing: "that is a first edition", "skip that shelf, it is not mine".
6. When the agent sees art, it asks: "is that an original or a print?"
7. Type "done" or tap **End sweep**. The background stages run: SAM spines, OCR, identification, checker, items, pricing, packet.
8. The agent reads back a short summary. The report link appears on screen.

Sweep order for this test: marker first, then shelf 1 top to bottom, then the table, then each wall, the floor, and
one ceiling corner. Hold each shelf row for about 2 seconds. Target: under 3 minutes.

## 5. Score (5 min)

```bash
uv run python -m eval.score sweeps/<id>/claim_packet.json ground_truth/ --md
```

Paste the table into `docs/RESULTS.md`. Report every bar, also the ones that fail.
