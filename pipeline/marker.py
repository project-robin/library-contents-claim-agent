"""Print-ready A4 PDF with two ArUco markers (ids 0 and 1) for metric scale.

    uv run python -m pipeline.marker   ->  assets/aruco_markers_A4.pdf
Print at 100% / "actual size", then measure a marker's black edge with a ruler and set MARKER_SIZE_CM in .env.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline import settings
from pipeline.geometry import ARUCO

DPI = 300


def make(path=Path("assets/aruco_markers_A4.pdf"), size_cm=settings.MARKER_SIZE_CM):
    W, H = int(21.0 / 2.54 * DPI), int(29.7 / 2.54 * DPI)
    side = int(size_cm / 2.54 * DPI)
    pages = []
    for k in (0, 1):  # one marker per page: two 15 cm markers do not fit on A4
        page = Image.new("L", (W, H), 255)
        y = (H - side) // 3
        page.paste(Image.fromarray(np.asarray(cv2.aruco.generateImageMarker(ARUCO, k, side, borderBits=1))),
                   ((W - side) // 2, y))
        ImageDraw.Draw(page).text(((W - side) // 2, y + side + 30),
                                  f"ArUco 4x4 id {k} - black square must measure {size_cm} cm. "
                                  "Stand it against the books or lay it on a shelf.", fill=0)
        pages.append(page)
    path.parent.mkdir(exist_ok=True)
    pages[0].save(path, resolution=DPI, save_all=True, append_images=pages[1:])
    return path


if __name__ == "__main__":
    print(make())
