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
    # head turned forward, beak pitched down, shortened and curved like the sheet's side view;
    # the sculpt's crest is trimmed off the back of the head
    "head_pitch": 22, "beak_scale": 0.8, "beak_bend": 6.0, "crest_trim": [15.0, 21.0],
    "neck_offset": [-0.9, 11.0, 22.0], "body_top": 52.0,
    # a deep neck (side view) that is narrower side to side, white throat in front, purple nape behind
    "neck_path": [[-26, 56], [-24, 68], [-18, 79], [-8, 86]], "neck_radius": [9.5, 9.5, 9.0, 7.0],
    "neck_xscale": 1.35, "nape_offset": 5.5,
    # upper body leaned back so it stands upright; wing tips and rump drawn out backwards
    "body_tilt": 13, "body_pivot": [-5, 21], "rump_stretch": [-8.0, 0.6, 18.0, 26.0, 48.0, 62.0],
    # tail: comes out from under the wing tips, shorter, narrower, turned so the fan shows from the side
    "tail_axis": [0, 0.643, 0.766], "tail_root": [0, 22, 27], "tail_root_target": [0.0, 8.0, 38.0],
    "tail_scale": [0.62, 0.6], "tail_roll": 90,
    # feet: brought in under the body, turned forward, back toes lowered onto the ground
    "leg_stretch": [6.0, 6.0, 16.0], "leg_narrow": 8.0,
    "foot_twist": -47, "foot_shift": 2.0, "foot_shift_zone": [3.0, 25.0],
    "hallux_drop": [-6.0, -1.0, 6.0, 10.0, 4.0],
    # long thin shins replace the sculpt's short thighs; the belly bottom is rounded off
    "shins": {"ankle": [11.4, -9.5, 9.8], "hip": [9.0, -5.0, 30.0], "radius": [2.7, 2.5], "blend": 1.5,
              "feet_top": 10.5, "belly": 24.0, "slab_back": 9.0, "belly_half": 20.0, "belly_rc": 14.0,
              "belly_soft": 2.5, "bevel": [[-34, 36], [-20, 24]], "rump_cut": [[6, 48], [26, 22], 52.0]},
}
# cheek tufts: a fan of three curled feathers on each side of the head (head-local, mm)
TUFTS = {"flatten": 1.7, "blend": 2.8, "feathers": [
    [[11.0, 13.0, -2.5, 3.2], [19.0, 16.5, 0.5, 2.5], [22.1, 17.6, 3.0, 1.6]],
    [[11.0, 13.0, -3.5, 3.2], [20.0, 16.5, -3.5, 2.6], [23.9, 17.2, -3.85, 1.7]],
    [[11.0, 13.0, -4.5, 3.0], [18.5, 16.5, -8.0, 2.4], [21.0, 17.2, -10.8, 1.5]]]}
PAINT = dict(
    SP.PAINT, brow_cap_z=9.0, lid_level=0.08,
    # bigger whites of the eyes, lids only over the eyeballs
    eye_grow=3.5, eye_back=0.9, lid_grow=1.0, lid_ring=1.4, lid_back=3.0,
    # body colors laid out on the side view (y, z in mm): the wing outline, the white band between
    # the purple coverts and the navy primaries (two points, half width, scallop depth and pitch),
    # and the white chest and belly in front of the back line
    wing_x=9.0, wing_wrap=[11.0, 1.6],
    wing_t=[[-21, 66], [-24.6, 58], [-24.6, 52], [-23, 47], [-20, 43.5], [-12, 36.4], [-2, 30.5], [12, 26],
            [24, 24], [28, 25], [14, 44], [3, 56], [-8, 68]],
    band_t=[[-19, 41], [9, 48.5], 2.3, 1.8, 6.0],
    chest_t=[[-60, 70], [-20, 70], [-20, 62], [-12, 50], [-3, 40], [6, 30], [12, 20], [12, 10], [-60, 10]],
)
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
