"""Build the multi-color print files from the sculpted STL.

    python build_print.py SCULPT.stl --out output --pitch 0.25

1. move the sculpt into its body frame and voxelize it
2. re-pose it to the reference sheet (stl_repose.py)
3. color its surface (paint.py)
4. split it into one closed, non-overlapping solid per color
5. check the parts and export STL per color, a combined 3MF and a Blender scene
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

import paint
import stl_repose as R

POSE = {
    # head turned forward, beak pitched down like the sheet's side view
    "head_pitch": 25, "neck_offset": [-0.9, 11.0, 22.0], "neck_radius": 6.6, "neck_via": [[0, -1.0, 12]],
    # upper body leaned back so it stands upright like the sheet
    "body_tilt": 13, "body_pivot": [-5, 21],
    # tail: steeper, shorter, narrower, and turned so the feather fan shows from the side
    "tail_axis": [0, 0.56, 0.83], "tail_root": [0, 22, 27], "tail_scale": [0.7, 0.6], "tail_roll": 90,
    # feet turned to point forward and moved back under the body
    "foot_twist": -47, "foot_shift": 7.0,
}
BOUNDS = ((-45.0, 45.0), (-78.0, 78.0), (0.0, 118.0))


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def label_volume(mesh, labels, axes):
    """Color of every voxel = color of the nearest surface vertex."""
    xs, ys, zs = axes
    pitch = float(xs[1] - xs[0])
    shape = (len(xs), len(ys), len(zs))
    idx = np.rint((np.asarray(mesh.vertices) - [xs[0], ys[0], zs[0]]) / pitch).astype(int)
    idx = np.clip(idx, 0, np.array(shape) - 1)
    seed = np.full(shape, -1, np.int8)
    seed[idx[:, 0], idx[:, 1], idx[:, 2]] = labels
    ind = ndimage.distance_transform_edt(seed < 0, return_distances=False, return_indices=True)
    vol = seed[ind[0], ind[1], ind[2]]
    return vol


def color_meshes(union, vol, axes, blur=0.8):
    xs, ys, zs = axes
    pitch = float(xs[1] - xs[0])
    origin = np.array([xs[0], ys[0], zs[0]]) - pitch
    out = {}
    for i, name in enumerate(paint.NAMES):
        ind = (vol == i)
        if not ind.any():
            continue
        g = ndimage.gaussian_filter(ind.astype(np.float32), blur)
        f = np.minimum(union, g)
        f = np.pad(f, 1, constant_values=0.0)
        if f.max() < 0.5:
            continue
        v, faces, _, _ = marching_cubes(f, 0.5, spacing=(pitch,) * 3)
        m = trimesh.Trimesh(v + origin, faces[:, ::-1], process=True)
        if m.volume < 0:
            m.invert()
        parts = m.split(only_watertight=False)
        keep = [p for p in parts if abs(p.volume) > 0.3]
        m = trimesh.util.concatenate(keep)
        out[name] = m
        log(f"  {name:7s} {len(m.faces):8d} tris  {m.volume / 1000:6.2f} cm3  bodies {len(keep):3d} "
            f"(dropped {len(parts) - len(keep)} specks)  watertight {m.is_watertight}")
    return out


def overlap_check(meshes):
    """Exact pairwise intersection volumes between the color parts (mm^3)."""
    import manifold3d as mf
    solids = {}
    for name, m in meshes.items():
        mesh = mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                       tri_verts=np.asarray(m.faces, np.uint32))
        solids[name] = mf.Manifold(mesh)
    names = list(solids)
    worst = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = (solids[a] ^ solids[b]).volume()
            if v > 0.01:
                worst[f"{a}/{b}"] = round(float(v), 3)
    return worst


def write_3mf(path, meshes):
    """One object made of one component per color, so slicers load a single multi-part object."""
    import zipfile
    hexcol = {n: "#%02X%02X%02X" % tuple(int(255 * c) for c in paint.PALETTE[n]) for n in meshes}
    parts = ['<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
             '<metadata name="Title">Bird, multi-color</metadata>\n<resources>\n'
             '<basematerials id="1">\n']
    parts += [f'<base name="{n}" displaycolor="{hexcol[n]}"/>\n' for n in meshes]
    parts.append('</basematerials>\n')
    for i, (n, m) in enumerate(meshes.items()):
        v = np.asarray(m.vertices)
        f = np.asarray(m.faces)
        parts.append(f'<object id="{i + 2}" name="bird_{n}" type="model" pid="1" pindex="{i}">\n<mesh>\n<vertices>\n')
        parts.append("".join(f'<vertex x="{a:.4f}" y="{b:.4f}" z="{c:.4f}"/>\n' for a, b, c in v))
        parts.append("</vertices>\n<triangles>\n")
        parts.append("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in f))
        parts.append("</triangles>\n</mesh>\n</object>\n")
    top = len(meshes) + 2
    parts.append(f'<object id="{top}" name="bird" type="model">\n<components>\n')
    parts += [f'<component objectid="{i + 2}"/>\n' for i in range(len(meshes))]
    parts.append(f'</components>\n</object>\n</resources>\n<build>\n<item objectid="{top}"/>\n</build>\n</model>\n')
    ctypes = ('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
              '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "".join(parts))


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

    log("loading sculpt")
    sculpt = trimesh.load(args.stl, force="mesh")
    sculpt.apply_transform(R.body_frame_matrix())
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    src_path = os.path.join(cache, f"source_{args.src_pitch}.npz")
    if os.path.exists(src_path):
        src = R.load_source(src_path)
    else:
        log(f"voxelizing at {args.src_pitch} mm")
        src = R.Source(sculpt, args.src_pitch)
        np.savez(src_path, field=src.field, lo=src.lo, pitch=src.pitch)

    import hashlib
    tag = hashlib.md5(json.dumps(POSE, sort_keys=True).encode()).hexdigest()[:8]
    union_path = os.path.join(cache, f"union_{args.pitch}_{tag}.npz")
    if os.path.exists(union_path):
        d = np.load(union_path)
        union, axes = d["union"], (d["xs"], d["ys"], d["zs"])
    else:
        log(f"re-posing at {args.pitch} mm")
        union, axes, _ = R.assemble(src, POSE, args.pitch, BOUNDS)
        lab, n = ndimage.label(union > 0.5)
        if n > 1:                                   # keep the bird, drop loose specks
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = ndimage.binary_dilation(lab == sizes.argmax(), iterations=2)
            union[~keep] = 0.0
            log(f"dropped {n - 1} loose speck(s)")
        del lab
        np.savez(union_path, union=union, xs=axes[0], ys=axes[1], zs=axes[2])
    whole = R.to_mesh(union, axes)
    log(f"re-posed solid: {len(whole.faces)} tris, watertight {whole.is_watertight}, "
        f"{whole.volume / 1000:.1f} cm3, size {np.round(whole.extents, 1).tolist()} mm")

    log("painting")
    pose = R.Pose(POSE)
    labels, _, _ = paint.paint(whole, src, pose, args.pitch, paint.SourceLabels(sculpt, src))
    del src

    log("splitting into color parts")
    vol = label_volume(whole, labels, axes)
    meshes = color_meshes(union, vol, axes)
    del vol

    log("checking overlaps between parts")
    ov = overlap_check(meshes)
    total = sum(ov.values())
    log(f"  total overlap between touching parts: {total:.3f} mm3 "
        f"({total / (whole.volume) * 100:.4f}% of the model); pairs: {ov}")

    log("exporting")
    whole.export(os.path.join(out, "stl", "bird_single_piece.stl"))
    for name, m in meshes.items():
        m.export(os.path.join(out, "stl", f"bird_{name}.stl"))
    write_3mf(os.path.join(out, "bird_all_colors.3mf"), meshes)
    colored = whole.copy()
    cols = (np.array([paint.PALETTE[n] for n in paint.NAMES]) * 255).astype(np.uint8)
    colored.visual.vertex_colors = np.c_[cols[labels], np.full(len(labels), 255, np.uint8)]
    colored.export(os.path.join(out, "bird_painted_preview.ply"))

    stats = {name: {"triangles": int(len(m.faces)), "volume_cm3": round(m.volume / 1000, 2),
                    "watertight": bool(m.is_watertight), "bodies": int(len(m.split(only_watertight=False)))}
             for name, m in meshes.items()}
    stats["_whole"] = {"size_mm": np.round(whole.extents, 1).tolist(), "volume_cm3": round(whole.volume / 1000, 1),
                       "centre_of_mass": np.round(whole.center_mass, 1).tolist(), "overlap": ov}
    json.dump(stats, open(os.path.join(out, "parts.json"), "w"), indent=1)

    if not args.no_blend:
        import blender_parts
        blender_parts.build(meshes, whole, out)
    log("done")


if __name__ == "__main__":
    main()
