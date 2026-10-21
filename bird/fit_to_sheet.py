"""Push the model's surface onto the sheet's outlines by moving its vertices.

The model's side and front outlines are compared with the sheet's SIDE and FRONT views at
the same height, standing on the same ground line. Every point of the model's outline is
moved onto the nearest point of the sheet's outline (only where the two outlines face the
same way and are close), those moves are spread smoothly across the whole view (a membrane
fit: the smoothest displacement that makes the outline moves), and every vertex of every
color part is moved by that same displacement. The parts therefore keep touching exactly,
keep their triangles, and stay watertight. The side view moves vertices in y and z, the
front view moves them in x (kept symmetric). This repeats a few times, re-measuring each time.

    python fit_to_sheet.py output_smooth --out output_fit
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
from PIL import Image
from scipy import ndimage
from skimage.measure import find_contours

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "ref", "reference_sheet.png")
BG = np.array([242.5, 239.0, 233.0])
G = 0.5                                                   # grid pitch, mm

# sheet views: crop (c0, r0, c1, r1), ground row, top row; model axes shown (horizontal, vertical)
VIEWS = {
    "side": dict(crop=(395, 120, 1150, 920), ground=888, top=150, h=1, span=(-80.0, 80.0)),
    "front": dict(crop=(20, 100, 420, 900), ground=887, top=130, h=0, span=(-50.0, 50.0)),
}
Z_SPAN = (-4.0, 126.0)


def sheet_mask():
    im = np.asarray(Image.open(SHEET).convert("RGB")).astype(float)
    fg = np.abs(im - BG).sum(2) > 40
    return ndimage.binary_fill_holes(ndimage.binary_closing(fg, iterations=2))


def grid_axes(view):
    h0, h1 = VIEWS[view]["span"]
    return np.arange(h0, h1 + G / 2, G), np.arange(Z_SPAN[0], Z_SPAN[1] + G / 2, G)


def ref_on_grid(fg, view, height, shift):
    """The sheet's silhouette for a view, resampled on the mm grid (rows z, cols horizontal)."""
    v = VIEWS[view]
    c0, r0, c1, r1 = v["crop"]
    s = (v["ground"] - v["top"]) / height
    hs, zs = grid_axes(view)
    H, Z = np.meshgrid(hs, zs)
    cols = (c0 + c1) / 2 + (H - shift) * s
    rows = v["ground"] - Z * s
    inside = (cols >= c0) & (cols < c1) & (rows >= r0) & (rows < r1)
    m = ndimage.map_coordinates(fg.astype(np.uint8), [rows, cols], order=0, mode="constant") > 0
    return m & inside


def model_on_grid(V, view):
    """The model's silhouette for a view: its vertices splatted on the mm grid, holes filled."""
    hs, zs = grid_axes(view)
    h = V[:, VIEWS[view]["h"]]
    i = np.round((V[:, 2] - zs[0]) / G).astype(int)
    j = np.round((h - hs[0]) / G).astype(int)
    ok = (i >= 0) & (i < len(zs)) & (j >= 0) & (j < len(hs))
    m = np.zeros((len(zs), len(hs)), bool)
    m[i[ok], j[ok]] = True
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def best_shift(fg, model, view, height, rng):
    best = None
    for s in rng:
        r = ref_on_grid(fg, view, height, s)
        q = iou(model, r)
        if best is None or q > best[0]:
            best = (q, s)
    return best


def signed_distance(mask):
    """mm; positive outside the mask, negative inside."""
    return (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)) * G


def outline_moves(model, ref, dmax, min_dot, frozen):
    """Points of the model's outline (grid units) and how far to move each (mm), smoothed along
    the outline. Points whose nearest sheet outline is far, or faces another way, or that lie in
    `frozen` (rows x cols bool), are pinned in place."""
    D = ndimage.gaussian_filter(signed_distance(ref), 1.0)
    gz, gh = np.gradient(D)
    soft = ndimage.gaussian_filter(model.astype(float), 1.5)
    mz, mh = np.gradient(soft)
    pts, moves = [], []
    for c in find_contours(model.astype(float), 0.5):
        if len(c) < 8:
            continue
        idx = [c[:, 0], c[:, 1]]
        d = ndimage.map_coordinates(D, idx, order=1)
        g = np.stack([ndimage.map_coordinates(gz, idx, order=1), ndimage.map_coordinates(gh, idx, order=1)], 1)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        n = -np.stack([ndimage.map_coordinates(mz, idx, order=1), ndimage.map_coordinates(mh, idx, order=1)], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        mv = -d[:, None] * g                                   # mm, (dz, dh)
        good = (np.abs(d) < dmax) & ((n * g).sum(1) > min_dot)
        good &= ~ndimage.map_coordinates(frozen.astype(np.uint8), idx, order=0).astype(bool)
        mv[~good] = 0.0
        # smooth along the outline (closed loop), about 4 mm
        w = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2)
        w /= w.sum()
        for k in range(2):
            mv[:, k] = ndimage.convolve1d(mv[:, k], w, mode="wrap")
        pts.append(c)
        moves.append(mv)
    return np.concatenate(pts), np.concatenate(moves)


def membrane(shape, pts, vals):
    """Smoothest field on the grid (harmonic) that takes the given values at the given points."""
    ny, nx = shape
    N = ny * nx
    idx = np.arange(N).reshape(shape)
    rows, cols, data = [], [], []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        rows += [a, b, a, b]
        cols += [b, a, a, b]
        data += [-np.ones(len(a)), -np.ones(len(a)), np.ones(len(a)), np.ones(len(a))]
    L = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    pi = np.clip(np.round(pts).astype(int), 0, [ny - 1, nx - 1])
    flat = pi[:, 0] * nx + pi[:, 1]
    cnt = np.bincount(flat, minlength=N)
    fixed = cnt > 0
    out = []
    for k in range(vals.shape[1]):
        v = np.bincount(flat, weights=vals[:, k], minlength=N)[fixed] / cnt[fixed]
        free = ~fixed
        A = L[free][:, free]
        rhs = -L[free][:, fixed] @ v
        u = np.zeros(N)
        u[fixed] = v
        u[free] = spla.spsolve(A.tocsc(), rhs)
        out.append(u.reshape(shape))
    return out


def sample(field, view, P):
    hs, zs = grid_axes(view)
    h = P[:, VIEWS[view]["h"]]
    return ndimage.map_coordinates(field, [(P[:, 2] - zs[0]) / G, (h - hs[0]) / G], order=1, mode="nearest")


def frozen_zones(view, H, Z):
    """Where the outline is left alone: legs and feet (both views); in the side view the head and
    beak (the sheet's beak is drawn open) and the tail tips (the sheet's three separate feathers
    do not match one fan); in the front view the head (drawn from a different angle)."""
    if view == "side":
        return (Z < 26.0) | ((Z > 66.0) & (H < 3.0)) | ((H > 22.0) & (Z > 84.0))
    return (Z < 26.0) | (Z > 66.0)


def fit(meshes, whole, iters=4, step=0.8, dmax=(5.0, 4.0), stiff=2.0, log=print):
    fg = sheet_mask()
    height = float(whole.vertices[:, 2].max())
    V_all = {k: m.vertices.copy() for k, m in meshes.items()}
    Vw = whole.vertices.copy()
    report = []
    for it in range(iters + 1):
        stats = {}
        for view in ("side", "front"):
            model = model_on_grid(Vw, view)
            rng = np.arange(-25, 25.01, 0.5) if view == "side" else np.arange(-6, 6.01, 0.5)
            q, shift = best_shift(fg, model, view, height, rng)
            stats[view] = round(float(q), 3)
            if it == iters:
                continue
            ref = ref_on_grid(fg, view, height, shift)
            hs, zs = grid_axes(view)
            Hh, Z = np.meshgrid(hs, zs)
            frozen = frozen_zones(view, Hh, Z)
            pts, mv = outline_moves(model, ref, dmax[0] if view == "side" else dmax[1], 0.3, frozen)
            # pin the far border and everything below the knees
            gy, gx = np.meshgrid(np.arange(len(zs)), np.arange(len(hs)), indexing="ij")
            border = (gy == 0) | (gy == len(zs) - 1) | (gx == 0) | (gx == len(hs) - 1)
            pins = np.argwhere(border | (Z < 24.0))
            pts = np.concatenate([pts, pins.astype(float)])
            mv = np.concatenate([mv, np.zeros((len(pins), 2))])
            dz, dh = membrane(model.shape, pts, mv * step)
            dz, dh = (ndimage.gaussian_filter(f, stiff / G) for f in (dz, dh))     # no sharp bends
            if view == "front":                             # keep it symmetric: dx(-x) = -dx(x)
                dh = 0.5 * (dh - dh[:, ::-1])
                moves = lambda P: np.stack([sample(dh, view, P), np.zeros(len(P)), np.zeros(len(P))], 1)
            else:
                moves = lambda P: np.stack([np.zeros(len(P)), sample(dh, view, P), sample(dz, view, P)], 1)
            Vw = Vw + moves(Vw)
            for k in V_all:
                V_all[k] = V_all[k] + moves(V_all[k])
        report.append(stats)
        log(f"  pass {it}: silhouette match with the sheet (IoU) {stats}")
    for k, m in meshes.items():
        m.vertices = V_all[k]
    whole.vertices = Vw
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="build folder with stl/bird_*.stl")
    ap.add_argument("--out", default="output_fit")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--no-blend", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "stl"), exist_ok=True)
    meshes = {}
    for p in sorted(glob.glob(os.path.join(args.folder, "stl", "bird_*.stl"))):
        name = os.path.basename(p)[5:-4]
        if name != "single_piece":
            meshes[name] = trimesh.load(p, force="mesh")
    whole = trimesh.load(os.path.join(args.folder, "stl", "bird_single_piece.stl"), force="mesh")
    report = fit(meshes, whole, iters=args.iters)

    from build_print import overlap_check, write_3mf
    ov = overlap_check(meshes)
    print(f"  total overlap between parts: {sum(ov.values()):.3f} mm3; pairs over 0.01 mm3: {ov}")
    whole.export(os.path.join(out, "stl", "bird_single_piece.stl"))
    for name, m in meshes.items():
        m.export(os.path.join(out, "stl", f"bird_{name}.stl"))
        print(f"  {name:7s} watertight {m.is_watertight}  {m.volume / 1000:6.2f} cm3")
    write_3mf(os.path.join(out, "bird_all_colors.3mf"), meshes)
    stats = {name: {"triangles": int(len(m.faces)), "volume_cm3": round(m.volume / 1000, 2),
                    "watertight": bool(m.is_watertight)} for name, m in meshes.items()}
    stats["_whole"] = {"size_mm": np.round(whole.extents, 1).tolist(), "volume_cm3": round(whole.volume / 1000, 1),
                       "centre_of_mass": np.round(whole.center_mass, 1).tolist(), "overlap_mm3": ov,
                       "silhouette_iou_per_pass": report}
    json.dump(stats, open(os.path.join(out, "parts.json"), "w"), indent=1)
    if not args.no_blend:
        import blender_parts
        blender_parts.build(meshes, whole, out)


if __name__ == "__main__":
    main()
