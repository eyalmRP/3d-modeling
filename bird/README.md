# Multi-color bird figure

A print-ready, multi-color version of the cartoon hornbill, re-posed to match the side view
of the turnaround sheet in `ref/reference_sheet.png`: standing upright, head facing forward
with the beak drooping, both wings folded, the three-feather tail fanned and raised behind.
The sculpt itself comes from a separate STL of the same character (not stored here), whose
original pose has the head turned back over its shoulder, one wing raised and a very long
tail. `repose_side_compare_v2.png` shows the sculpt before and after next to the sheet.

## Output (`output/`, generated)

| File | What it is |
| --- | --- |
| `stl/bird_<color>.stl` | One closed solid per color, all in the same coordinates |
| `bird_all_colors.3mf` | All color parts in one file, already positioned |
| `stl/bird_single_piece.stl` | The re-posed figure as one solid (for single-color prints) |
| `bird.blend` | Blender scene: one object and material per color, in mm |
| `bird_painted_preview.ply` | The re-posed surface with vertex colors |
| `render_*.png`, `preview_sheet.png`, `compare_with_sheet.png` | Previews |
| `parts.json` | Volume, triangle count and checks for every part |

Colors: purple (head, back, wings, tail base), navy (wing primaries, tail tips, back-of-head
tuft), white (chest, throat, eyes, wing and tail bands), red (beak and jaw), yellow (beak
patches), orange (legs and feet), light blue (eyelids), black (brows and pupils).

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
* **Size:** 100% scale is about 113 mm tall, 82 mm wide and 143 mm from beak tip to tail tip.
  Scale the whole object, never individual parts.
* **Orientation:** as exported, standing on its feet.
* **Supports:** needed under the beak, the tail, the wing tips, the belly and the brows. Tree
  (organic) supports touching the build plate only are a good start; a support interface in a
  contrasting material helps where it touches the beak.
* **Stability:** the centre of mass sits about 7 mm inside the footprint of the toes. It stands
  on its own; for extra safety glue it to a small base.

## Rebuilding

Requires Python 3.11 with `bpy` (Blender as a module), `trimesh`, `scikit-image`, `scipy`,
`matplotlib`, `manifold3d` and `libigl`.

```
python build_smooth.py path/to/sculpt.stl --out output_smooth --pitch 0.25
```

* `sdf_source.py` turns the sculpt into an exact signed-distance field (libigl), so the
  re-posed surface stays as smooth as the original sculpt.
* `stl_repose.py` holds the re-pose transforms: the sculpt is aligned to its own symmetry
  plane, the half with the folded wing is mirrored (removing the raised wing and evening out
  the legs), the upper body is leaned back to stand upright, the head is turned forward with a
  shorter, more curved beak on a new slimmer neck, the tail is re-aimed, shortened and turned so
  its feather fan shows from the side, and the legs are lengthened with the feet turned forward.
* `sdf_pipeline.py` samples each piece through its transform, joins the pieces with smooth
  (filleted) unions, adds the cheek tufts, and paints with smooth color fields (eyeball spheres,
  lid caps, brow arcs, beak crease, wing outline and bands, tail bands), so color borders are
  clean curves. Each color's solid is the part of the figure where that color's field wins.
* `build_smooth.py` holds the pose, tuft and paint settings, extracts the parts, checks them
  and exports. `blender_parts.py` builds the Blender scene and the preview renders.
* `check_parts.py` reports watertightness and thin pieces of exported STLs.
