"""Read the reference turnaround sheet: silhouettes and per-pixel color classes per view."""
import os

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "ref", "reference_sheet.png")

BG = np.array([242.5, 239.0, 233.0])

# reference colors sampled from the sheet (k-means of the figure pixels)
CLASSES = {
    "purple": [(108, 100, 160), (93, 86, 147), (125, 118, 175)],
    "navy": [(34, 24, 85), (57, 47, 107)],
    "white": [(249, 248, 246), (235, 234, 236)],
    "red": [(237, 55, 56), (207, 40, 29), (225, 30, 35)],
    "yellow": [(251, 171, 43), (245, 150, 40)],
    "blue": [(179, 193, 221), (150, 175, 225)],
    "black": [(13, 13, 15), (40, 38, 45)],
    "pink": [(216, 106, 99), (230, 140, 135)],
    "line": [(165, 156, 162), (127, 107, 103), (121, 16, 11)],
}
NAMES = list(CLASSES)

# column ranges of the three views on the sheet, and label rows to ignore
VIEWS = {"front": (30, 400), "side": (400, 1145), "top": (1150, 1410)}
LABEL_ROW = {"front": 940, "side": 940, "top": 1005}


def load():
    return np.asarray(Image.open(SHEET).convert("RGB")).astype(np.float32)


def silhouette(img, view):
    c0, c1 = VIEWS[view]
    sub = img[:LABEL_ROW[view], c0:c1]
    fg = np.abs(sub - BG).sum(2) > 40
    fg = ndimage.binary_closing(fg, iterations=2)
    fg = ndimage.binary_fill_holes(fg)
    lab, n = ndimage.label(fg)
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes > 2000)[0])
    out = np.zeros(img.shape[:2], bool)
    out[:LABEL_ROW[view], c0:c1] = keep
    return out


def classify(img, mask):
    """Nearest-palette class per pixel inside mask; thin outline strokes are replaced
    by the surrounding fill so they don't become their own color."""
    best = np.full(img.shape[:2], 1e9, np.float32)
    cls = np.zeros(img.shape[:2], np.int16)
    for i, name in enumerate(NAMES):
        for rgb in CLASSES[name]:
            d = ((img - np.array(rgb, np.float32)) ** 2).sum(2)
            better = d < best
            best[better] = d[better]
            cls[better] = i
    cls[~mask] = -1
    line = NAMES.index("line")
    solid = np.zeros_like(mask)
    for i, name in enumerate(NAMES):
        if name == "line":
            continue
        m = ndimage.binary_opening(cls == i, iterations=2 if name not in ("black",) else 1)
        solid |= m
    known = solid & mask & (cls != line)
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    filled = cls[idx[0], idx[1]]
    filled[~mask] = -1
    return filled


def analyse():
    img = load()
    masks = {v: silhouette(img, v) for v in VIEWS}
    full = np.zeros(img.shape[:2], bool)
    for m in masks.values():
        full |= m
    cls = classify(img, full)
    return img, masks, cls


if __name__ == "__main__":
    img, masks, cls = analyse()
    pal = {"purple": (108, 100, 160), "navy": (34, 24, 85), "white": (255, 255, 255), "red": (237, 55, 56),
           "yellow": (251, 171, 43), "blue": (179, 193, 221), "black": (0, 0, 0), "pink": (216, 106, 99),
           "line": (0, 255, 0)}
    vis = np.full(img.shape, 200, np.uint8)
    for i, n in enumerate(NAMES):
        vis[cls == i] = pal[n]
    Image.fromarray(vis).save(os.path.join(os.environ.get("OUT", HERE), "classified.png"))
    for v, m in masks.items():
        ys, xs = np.where(m)
        print(v, "rows", ys.min(), ys.max(), "cols", xs.min(), xs.max())
