# Multi-color bird figure

A print-ready, multi-color version of the cartoon hornbill, re-posed to match the side view
of the turnaround sheet in `ref/reference_sheet.png`: standing upright, head facing forward
with the beak drooping, both wings folded, the three-feather tail fanned and raised behind.
The sculpt itself comes from a separate STL of the same character (not stored here), whose
original pose has the head turned back over its shoulder, one wing raised and a very long
tail. `repose_side_compare_v2.png` shows the sculpt before and after next to the sheet.

## Output (the `--out` folder, generated)

| File | What it is |
| --- | --- |
| `stl/bird_<color>.stl` | One closed solid per color, all in the same coordinates |
| `bird_all_colors.3mf` | All color parts in one file, already positioned |
| `stl/bird_single_piece.stl` | The re-posed figure as one solid (for single-color prints) |
| `bird.blend` | Blender scene: one object and material per color, in mm |
| `render_*.png`, `preview_sheet.png`, `compare_with_sheet.png` | Previews |
| `parts.json` | Volume, triangle count and checks for every part |

Colors: purple (head, cheek tufts, nape, back, wings, tail), navy (wing primaries, tail tips),
white (chest, throat, eyes, wing and tail bands), red (beak and jaw), yellow (beak patches),
orange (legs and feet), light blue (eyelids), black (brows and pupils).

The parts tile the figure exactly: they touch along shared surfaces but do not overlap
(total overlap between all parts is a fraction of a cubic millimetre). Each part is
watertight.

## Printing

* **Multi-material (AMS / MMU / toolchanger):** open `bird_all_colors.3mf`, or select all the
  `bird_<color>.stl` files and import them together; when the slicer asks whether to load them
  as a single object with multiple parts, answer **yes**. Then assign a filament to each part.
  Bambu Studio, OrcaSlicer and PrusaSlicer all work this way. Eight colors are used; if you have
  fewer slots, combine parts that can share a filament (e.g. orange feet with the red beak,
  navy with black).
* **Size:** 100% scale is about 119 mm tall, 58 mm wide and 117 mm from beak tip to tail tip.
  Scale the whole object, never individual parts.
* **Orientation:** as exported, standing on its feet.
* **Supports:** needed under the beak, the tail, the wing tips, the belly and the brows. Tree
  (organic) supports touching the build plate only are a good start; a support interface in a
  contrasting material helps where it touches the beak.
* **Legs:** the shins are about 5 mm thick, like the sheet's thin legs. Print them solid (more
  walls or a 100% infill modifier on the orange part) and keep the supports clear of them.
* **Stability:** the back toes rest on the ground, and the centre of mass sits about 17 mm
  inside the footprint of the toes. It stands on its own; for extra safety glue it to a base.

## Rebuilding

Requires Python 3.11 with `bpy` (Blender as a module), `trimesh`, `scikit-image`, `scipy`,
`matplotlib`, `manifold3d` and `libigl`.

```
python build_smooth.py path/to/sculpt.stl --out output_smooth --pitch 0.25
python fit_to_sheet.py output_smooth --out output_fit
```

* `sdf_source.py` turns the sculpt into an exact signed-distance field (libigl), so the
  re-posed surface stays as smooth as the original sculpt.
* `stl_repose.py` holds the re-pose transforms: the sculpt is aligned to its own symmetry
  plane, the half with the folded wing is mirrored (removing the raised wing and evening out
  the legs), the upper body is leaned back to stand upright and the wing tips are drawn out
  backwards, the head is turned forward with a shorter, more curved beak, the tail is re-aimed,
  shortened and turned so its feather fan shows from the side, and the feet are brought in
  under the body and turned forward with the back toes on the ground.
* `sdf_pipeline.py` samples each piece through its transform and joins the pieces with smooth
  (filleted) unions. It adds new geometry where the sculpt differs from the sheet: a deep neck
  with a white throat, long thin shins in place of the sculpt's short thighs (with the belly
  bottom rounded off), and cheek tufts. Colors are smooth fields (eyeball spheres, lid caps,
  brow arcs, beak crease, the wing outline with its scalloped white band laid out on the side
  view, tail bands), so color borders are clean curves. Each color's solid is the part of the
  figure where that color's field wins.
* `build_smooth.py` holds the pose, tuft and paint settings, extracts the parts, checks them
  and exports. `blender_parts.py` builds the Blender scene and the preview renders.
* `fit_to_sheet.py` then moves the vertices of all the color parts so the figure's side and
  front outlines land on the sheet's. Each outline point is moved onto the nearest sheet
  outline point that faces the same way, the moves are spread smoothly over the whole figure,
  and every part is moved by the same displacement, so the parts still fit together exactly and
  stay watertight. The head and beak, the tail tips, and the legs and feet are left as they
  are: they already match, or the sheet draws them in a way a single solid cannot follow
  (open beak, three separate tail feathers, two legs side by side).
  `before_now_vs_sheet.png` and `closeups_before_now.png` show the result.
* `check_parts.py` reports watertightness and thin pieces of exported STLs.
