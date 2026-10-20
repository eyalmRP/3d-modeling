"""Printability checks on exported part STLs: watertightness, pieces, and thin sections.

    python check_parts.py output/stl --pitch 0.15
"""
import argparse
import glob
import os

import numpy as np
import trimesh
from scipy import ndimage

import stl_repose as R


def thickness_report(mesh, pitch):
    """Per connected piece: volume and the largest inscribed diameter (a piece whose largest
    inscribed diameter is small is thin everywhere and may not print)."""
    occ, lo = R.voxelize(mesh, pitch, margin=2 * pitch)
    lab, n = ndimage.label(occ)
    dist = ndimage.distance_transform_edt(occ) * pitch
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        vol = sel.sum() * pitch ** 3
        if vol < 0.05:
            continue
        idx = np.argwhere(sel).mean(0) * pitch + lo
        out.append((vol, 2 * dist[sel].max(), idx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pitch", type=float, default=0.15)
    ap.add_argument("--min", type=float, default=1.0, help="flag pieces thinner than this (mm)")
    args = ap.parse_args()
    for path in sorted(glob.glob(os.path.join(args.folder, "bird_*.stl"))):
        m = trimesh.load(path, force="mesh")
        pieces = thickness_report(m, args.pitch)
        thin = [p for p in pieces if p[1] < args.min]
        print(f"{os.path.basename(path):26s} watertight={m.is_watertight} volume={m.volume / 1000:6.2f} cm3 "
              f"pieces={len(pieces):3d} thinnest piece={min(p[1] for p in pieces):.2f} mm")
        for vol, d, c in sorted(thin, key=lambda p: p[1]):
            print(f"    thin piece: {d:.2f} mm across, {vol:.2f} mm3, near {np.round(c, 1).tolist()}")


if __name__ == "__main__":
    main()
