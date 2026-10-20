"""Build the multi-part bird: sample the SDF, mesh each color, assemble in Blender.

Run with the Python that has `bpy` installed:
    python build.py --voxel 0.25 --out output            # final quality
    python build.py --voxel 0.8 --out preview --no-export  # quick look
"""
import argparse
import os
import sys
import time

import numpy as np
from skimage.measure import marching_cubes
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bird_sdf  # noqa: E402


def sample(voxel):
    (x0, x1), (y0, y1), (z0, z1) = bird_sdf.BOUNDS
    xs = np.arange(x0, x1 + voxel, voxel, dtype=np.float32)
    ys = np.arange(y0, y1 + voxel, voxel, dtype=np.float32)
    zs = np.arange(z0, z1 + voxel, voxel, dtype=np.float32)
    fields = {c: np.empty((len(xs), len(ys), len(zs)), np.float32) for c in bird_sdf.COLORS}
    step = max(1, int(4e6 // (len(xs) * len(ys))))
    for k in range(0, len(zs), step):
        X, Y, Z = np.meshgrid(xs, ys, zs[k:k + step], indexing="ij")
        for c, f in bird_sdf.evaluate(X, Y, Z).items():
            fields[c][:, :, k:k + step] = f
    return fields, (xs[0], ys[0], zs[0])


def mesh_color(field, origin, voxel, blur):
    if blur > 0:
        field = ndimage.gaussian_filter(field, blur)
    field = np.pad(field, 1, constant_values=1e3)
    if field.min() >= 0:
        return None
    v, f, _, _ = marching_cubes(field, 0.0, spacing=(voxel,) * 3, method="lewiner")
    v += np.asarray(origin, np.float32) - voxel
    return v.astype(np.float32), f.astype(np.int32)


def drop_specks(v, f, min_volume):
    """Remove disconnected shells smaller than min_volume mm^3."""
    import trimesh
    m = trimesh.Trimesh(v, f, process=True)
    parts = m.split(only_watertight=False)
    keep = [p for p in parts if abs(p.volume) >= min_volume]
    dropped = len(parts) - len(keep)
    m = trimesh.util.concatenate(keep)
    return m, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--blur", type=float, default=0.6, help="gaussian blur of fields, in voxels")
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=24)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    t = time.time()
    fields, origin = sample(args.voxel)
    print(f"sampled {next(iter(fields.values())).shape} in {time.time() - t:.0f}s", flush=True)

    meshes = {}
    for c, field in fields.items():
        r = mesh_color(field, origin, args.voxel, args.blur)
        if r is None:
            print(f"  {c}: empty")
            continue
        m, dropped = drop_specks(*r, min_volume=0.5)
        meshes[c] = m
        print(f"  {c}: {len(m.faces)} tris, {m.volume / 1000:.2f} cm3, watertight={m.is_watertight}, "
              f"bodies={len(m.split(only_watertight=False))}, dropped specks={dropped}", flush=True)
    del fields

    import blender_scene
    blender_scene.build(meshes, out, render=not args.no_render, export=not args.no_export,
                        samples=args.samples)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
