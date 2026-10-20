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
* **Size:** 100% scale is about 107 mm tall (78 x 144 mm footprint including beak and tail).
  Scale the whole object, never individual parts.
* **Orientation:** as exported, standing on its feet.
* **Supports:** needed under the beak, the tail, the wing tips, the belly and the brows. Tree
  (organic) supports touching the build plate only are a good start; a support interface in a
  contrasting material helps where it touches the beak.
* **Stability:** the centre of mass sits about 7 mm inside the footprint of the toes. It stands
  on its own; for extra safety glue it to a small base.

## Rebuilding

Requires Python 3.11 with `bpy` (Blender as a module), `trimesh`, `scikit-image`, `scipy`,
`matplotlib` and `manifold3d`.

```
python build_print.py path/to/sculpt.stl --out output --pitch 0.25
```

* `stl_repose.py` aligns the sculpt to its own symmetry plane, voxelizes it, then cuts it into
  rigid pieces and re-poses them: the half with the folded wing is mirrored (removing the raised
  wing and evening out the legs), the upper body is leaned back to stand upright, the head is
  turned forward onto a new neck, the tail is re-aimed, shortened and turned so its feather fan
  shows from the side, and the feet are turned to point forward. The pose settings are the
  `POSE` dictionary at the top of `build_print.py`.
* `paint.py` colors the surface with rules written in each piece's own frame (eyeball spheres,
  lid caps, beak crease, wing outline and bands, tail bands, etc.), computed on the original
  sculpt and carried through the re-pose.
* `build_print.py` turns the colored surface into one solid per color, checks them and exports.
* `blender_parts.py` builds the Blender scene and the preview renders.
