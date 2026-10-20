"""Blender scene for the color parts: one object and material per color, preview renders."""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

import paint

HERE = os.path.dirname(os.path.abspath(__file__))


def _mesh_object(name, mesh, collection, mat=None):
    me = bpy.data.meshes.new(name)
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int32)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.ravel())
    me.loops.add(f.size)
    me.loops.foreach_set("vertex_index", f.ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3, dtype=np.int32))
    me.update()
    me.validate()
    me.polygons.foreach_set("use_smooth", np.ones(len(f), bool))
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def _material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def _render(scene, path, direction, target, scale, res=(900, 900)):
    d = Vector(direction).normalized()
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.location = Vector(target) + d * 500
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sun = scene.objects["Key"]
    sun.rotation_euler = (-d + Vector((0.35, 0.25, -0.7))).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)


def build(meshes, whole, out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001            # 1 Blender unit = 1 mm
    scene.unit_settings.length_unit = "MILLIMETERS"
    parts = bpy.data.collections.new("Color parts")
    scene.collection.children.link(parts)
    for name, m in meshes.items():
        _mesh_object(f"bird_{name}", m, parts, _material(name, paint.PALETTE[name]))
    ref = bpy.data.collections.new("Single piece (reference)")
    scene.collection.children.link(ref)
    ob = _mesh_object("bird_single_piece", whole, ref)
    ob.hide_render = True
    ob.hide_set(True)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -0.45
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.96, 0.95, 0.92, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.75
    sun = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "SUN"))
    sun.data.energy = 2.6
    scene.collection.objects.link(sun)

    bb = np.asarray(whole.bounds)
    c = bb.mean(0)
    size = float((bb[1] - bb[0]).max()) * 1.08
    renders = {}
    for name, d in (("front", (0, -1, 0)), ("side", (1, 0, 0)), ("top", (0, 0, 1)),
                    ("three_quarter", (0.75, -1.0, 0.45)), ("back_three_quarter", (-0.8, 1.0, 0.45))):
        path = os.path.join(out, f"render_{name}.png")
        dd = d if name != "top" else (0, -0.001, 1)
        _render(scene, path, dd, c, size)
        renders[name] = path

    from PIL import Image
    ims = [Image.open(renders[k]) for k in ("front", "side", "three_quarter", "back_three_quarter")]
    sheet = Image.new("RGB", (900 * 4, 900), (245, 242, 235))
    for i, im in enumerate(ims):
        sheet.paste(im, (900 * i, 0))
    sheet.save(os.path.join(out, "preview_sheet.png"))

    # side by side with the reference sheet's front and side views, same height
    refimg = Image.open(os.path.join(HERE, "ref", "reference_sheet.png")).convert("RGB")
    row = []
    for crop, key in (((20, 110, 410, 900), "front"), ((395, 130, 1150, 910), "side")):
        r = refimg.crop(crop)
        r = r.resize((int(r.width * 760 / r.height), 760))
        m = Image.open(renders[key]).convert("RGB")
        m = m.crop(m.getbbox() if False else (0, 0, m.width, m.height)).resize((760, 760))
        row += [r, m]
    canvas = Image.new("RGB", (sum(i.width for i in row), 760), (255, 255, 255))
    x = 0
    for im in row:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(os.path.join(out, "compare_with_sheet.png"))

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "bird.blend"))
