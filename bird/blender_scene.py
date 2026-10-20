"""Assemble the color parts in Blender, render previews and export print files."""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

import bird_sdf


def _clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def _object(name, mesh, mat, collection):
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
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def _camera(name, location, target, ortho_scale):
    cam = bpy.data.cameras.new(name)
    cam.type = "ORTHO"
    cam.ortho_scale = ortho_scale
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = location
    direction = Vector(target) - Vector(location)
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return ob


def _render_views(out, samples):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 0.98, 0.93, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7

    sun = bpy.data.lights.new("Key", "SUN")
    sun.energy = 3.5
    sun_ob = bpy.data.objects.new("Key", sun)
    scene.collection.objects.link(sun_ob)
    sun_ob.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))

    views = {
        "front": ((0, -300, 50), (0, 0, 50), 108),
        "side": ((300, 13, 50), (0, 13, 50), 120),
        "top": ((0, 13, 300), (0, 13, 0), 120),
        "three_quarter": ((-210, -210, 140), (0, 10, 48), 125),
    }
    paths = []
    for name, (loc, target, scale) in views.items():
        cam = _camera(name, loc, target, scale)
        if name == "top":
            cam.rotation_euler = (0, 0, math.pi)
        scene.camera = cam
        scene.render.resolution_x = 760
        scene.render.resolution_y = 760
        path = os.path.join(out, f"render_{name}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        paths.append(path)

    from PIL import Image
    ims = [Image.open(p) for p in paths]
    sheet = Image.new("RGB", (sum(i.width for i in ims), max(i.height for i in ims)), (237, 232, 219))
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width
    sheet.save(os.path.join(out, "preview_sheet.png"))
    return paths


def build(meshes, out, render=True, export=True, samples=24):
    import trimesh

    _clear()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001   # 1 Blender unit = 1 mm
    scene.unit_settings.length_unit = "MILLIMETERS"
    coll = bpy.data.collections.new("Bird parts")
    scene.collection.children.link(coll)

    for c, m in meshes.items():
        rgb, _ = bird_sdf.COLORS[c]
        _object(f"part_{c}", m, _material(c, rgb), coll)

    if render:
        _render_views(out, samples)

    if export:
        stl_dir = os.path.join(out, "stl")
        os.makedirs(stl_dir, exist_ok=True)
        for c, m in meshes.items():
            m.export(os.path.join(stl_dir, f"bird_{c}.stl"))
        tm_scene = trimesh.Scene()
        for c, m in meshes.items():
            rgb, _ = bird_sdf.COLORS[c]
            mm = m.copy()
            mm.visual.face_colors = [int(255 * v) for v in rgb] + [255]
            tm_scene.add_geometry(mm, node_name=f"bird_{c}", geom_name=f"bird_{c}")
        tm_scene.export(os.path.join(out, "bird_all_colors.3mf"))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "bird.blend"))
