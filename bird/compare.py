"""Project the model's silhouettes back onto the reference sheet and score the overlap."""
import numpy as np
from PIL import Image

import ref_model as rm


def project_to_sheet(sheet, grid, occ):
    xs, ys, zs = grid
    v = xs[1] - xs[0]
    H, W = sheet.cls.shape
    rows, cols = np.mgrid[0:H, 0:W].astype(float)
    out = np.zeros((H, W), bool)

    def idx(vals, axis):
        i = np.rint((vals - axis[0]) / v).astype(int)
        return i, (i >= 0) & (i < len(axis))

    # front: along y
    pf = occ.any(1)
    x = (cols - sheet.cx) * rm.S
    z = (rm.GROUND - rm.interp(rows, rm.F_ROWS, rm.F_TO_S)) * rm.S
    ix, okx = idx(x, xs)
    iz, okz = idx(z, zs)
    m = okx & okz & (cols >= 30) & (cols < 400)
    out[m] |= pf[ix[m], iz[m]]
    # side: along x
    ps = occ.any(0)
    y = (cols - rm.Y0) * rm.S
    z = (rm.GROUND - rows) * rm.S
    iy, oky = idx(y, ys)
    iz, okz = idx(z, zs)
    m = oky & okz & (cols >= 400) & (cols < 1145)
    out[m] |= ps[iy[m], iz[m]]
    # top: along z
    pt = occ.any(2)
    x = (cols - sheet.tx) * rm.S * rm.TOP_K
    y = (rm.interp(rows, rm.T_ROWS, rm.T_TO_S) - rm.Y0) * rm.S
    ix, okx = idx(x, xs)
    iy, oky = idx(y, ys)
    m = okx & oky & (cols >= 1150) & (cols < 1410)
    out[m] |= pt[ix[m], iy[m]]
    return out


def overlay(sheet, model_sil, path):
    ref = np.zeros_like(model_sil)
    for m in sheet.masks.values():
        ref |= m
    img = sheet.img.copy() * 0.35 + 255 * 0.65
    both = ref & model_sil
    img[both] = img[both] * 0.5 + np.array([120, 120, 120]) * 0.5
    img[model_sil & ~ref] = (230, 40, 40)      # model sticks out
    img[ref & ~model_sil] = (40, 90, 230)      # model is missing
    Image.fromarray(img.astype(np.uint8)).save(path)
    scores = {}
    for v, (c0, c1) in {"front": (30, 400), "side": (400, 1145), "top": (1150, 1410)}.items():
        r = ref[:, c0:c1]
        mm = model_sil[:, c0:c1]
        scores[v] = (r & mm).sum() / max((r | mm).sum(), 1)
    return scores
