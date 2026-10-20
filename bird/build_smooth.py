"""Build the multi-color print files from the sculpted STL, with smooth surfaces.

    python build_smooth.py SCULPT.stl --out output --pitch 0.25

1. move the sculpt into its body frame and compute its exact signed-distance field
2. re-pose it to the reference sheet and paint it, all as smooth fields (sdf_pipeline.py)
3. extract one closed, non-overlapping solid per color
4. check the parts and export STL per color, a combined 3MF and a Blender scene
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from skimage.measure import marching_cubes

import paint
import sdf_pipeline as SP
import sdf_source
import stl_repose as R
from build_print import overlap_check, write_3mf

POSE = {
    # head turned forward, beak pitched down, shortened and curved like the sheet's side view
    "head_pitch": 25, "beak_scale": 0.8, "beak_bend": 5.0,
    # a slimmer, slightly tapered neck rising into the back of the head
    "neck_offset": [-0.9, 11.0, 22.0], "neck_radius": [5.4, 4.8, 4.8], "neck_via": [[0, -1.0, 12]],
    "body_top": 52.0,
    # upper body leaned back so it stands upright
    "body_tilt": 13, "body_pivot": [-5, 21],
    # tail: steeper, shorter, narrower, and turned so the feather fan shows from the side
    "tail_axis": [0, 0.5, 0.866], "tail_root": [0, 22, 27], "tail_scale": [0.7, 0.6], "tail_roll": 90,
    # longer legs, brought in under the body; feet turned forward and set back
    "leg_stretch": [6.0, 6.0, 16.0], "leg_narrow": 5.0,
    "foot_twist": -47, "foot_shift": 10.0, "foot_shift_zone": [3.0, 25.0],
}
# cheek tufts: a fan of three curled feathers on each side of the head (head-local, mm)
TUFTS = {"flatten": 1.7, "blend": 2.2, "feathers": [
    [[11.0, 13.0, -2.5, 3.2], [19.0, 16.5, 0.5, 2.5], [23.5, 18.0, 4.0, 1.5]],
    [[11.0, 13.0, -3.5, 3.2], [20.0, 16.5, -3.5, 2.6], [25.5, 17.5, -4.0, 1.6]],
    [[11.0, 13.0, -4.5, 3.0], [18.5, 16.5, -8.0, 2.4], [22.0, 17.5, -12.0, 1.4]]]}
PAINT = dict(SP.PAINT, brow_cap_z=9.0, lid_level=0.08)
BOUNDS = ((-45.0, 45.0), (-80.0, 80.0), (0.0, 125.0))


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def extract(field, axes, min_volume=0.3):
    xs, ys, zs = axes
    pitch = float(xs[1] - xs[0])
    inside = np.argwhere(field < 0)
    if len(inside) == 0:
        return None
    lo = np.maximum(inside.min(0) - 2, 0)
    hi = np.minimum(inside.max(0) + 3, np.array(field.shape))
    sub = np.pad(field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 1, constant_values=1.0)
    v, f, _, _ = marching_cubes(sub, 0.0, spacing=(pitch,) * 3)
    v += np.array([xs[lo[0]], ys[lo[1]], zs[lo[2]]]) - pitch
    m = trimesh.Trimesh(v, f, process=True)
    parts = m.split(only_watertight=False)
    keep = [p for p in parts if abs(p.volume) > min_volume]
    return trimesh.util.concatenate(keep), len(parts) - len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--out", default="output")
    ap.add_argument("--pitch", type=float, default=0.25)
    ap.add_argument("--src-pitch", type=float, default=0.25)
    ap.add_argument("--no-blend", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "stl"), exist_ok=True)
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)

    src_path = os.path.join(cache, f"sdf_{args.src_pitch}.npz")
    if os.path.exists(src_path):
        src = sdf_source.SDFSource.load(src_path)
    else:
        log("loading sculpt and computing its exact distance field")
        sculpt = trimesh.load(args.stl, force="mesh")
        sculpt.apply_transform(R.body_frame_matrix())
        sculpt, _ = sdf_source.clean_mesh(sculpt)
        src, _ = sdf_source.build(sculpt, pitch=args.src_pitch)
        src.save(src_path)

    log(f"re-posing and painting at {args.pitch} mm")
    U, G, axes, pose = SP.assemble(src, POSE, args.pitch, BOUNDS, PAINT, TUFTS, log=log)
    del src

    log("extracting the figure and the color parts")
    whole, _ = extract(U, axes)
    log(f"  figure: {len(whole.faces)} tris, watertight {whole.is_watertight}, {whole.volume / 1000:.1f} cm3, "
        f"size {np.round(whole.extents, 1).tolist()} mm")
    meshes = {}
    for c in paint.NAMES:
        r = extract(np.maximum(U, G[c]), axes)
        if r is None:
            continue
        m, dropped = r
        meshes[c] = m
        log(f"  {c:7s} {len(m.faces):8d} tris  {m.volume / 1000:6.2f} cm3  "
            f"bodies {len(m.split(only_watertight=False)):3d} (dropped {dropped} specks)  watertight {m.is_watertight}")
        del G[c]

    log("checking overlaps between parts")
    ov = overlap_check(meshes)
    log(f"  total overlap between parts: {sum(ov.values()):.3f} mm3; pairs over 0.01 mm3: {ov}")

    log("exporting")
    whole.export(os.path.join(out, "stl", "bird_single_piece.stl"))
    for name, m in meshes.items():
        m.export(os.path.join(out, "stl", f"bird_{name}.stl"))
    write_3mf(os.path.join(out, "bird_all_colors.3mf"), meshes)
    stats = {name: {"triangles": int(len(m.faces)), "volume_cm3": round(m.volume / 1000, 2),
                    "watertight": bool(m.is_watertight), "bodies": int(len(m.split(only_watertight=False)))}
             for name, m in meshes.items()}
    stats["_whole"] = {"size_mm": np.round(whole.extents, 1).tolist(), "volume_cm3": round(whole.volume / 1000, 1),
                       "centre_of_mass": np.round(whole.center_mass, 1).tolist(), "overlap_mm3": ov}
    json.dump(stats, open(os.path.join(out, "parts.json"), "w"), indent=1)
    if not args.no_blend:
        import blender_parts
        blender_parts.build(meshes, whole, out)
    log("done")


if __name__ == "__main__":
    main()
