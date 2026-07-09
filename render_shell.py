"""render_shell.py — dollhouse room-shell builder for the audit renderer.

Builds a neutral floor (opaque) + walls with door/window openings cut out, and
applies a backface-culling material to the walls so the camera-facing (near)
wall faces render transparent (per surface normal) while far walls stay opaque.

This is the standard "dollhouse" convention: the camera always sees into the
room, and the far walls — with their door/window openings — remain visible as
architectural context. At the top-down default (pitch 0) the walls are edge-on
and contribute negligibly; as pitch increases the near walls fall away
automatically (their camera-facing faces become transparent). Doors and windows
are retained throughout: they are openings in the wall quad mesh, not discarded.

Walls are flat double-sided quads (no thickness): an opening is simply a hole in
the quad mesh, so a far wall shows its door/window as a see-through rectangular
opening. The backface-culling material (Blender `Backfacing` geometry node)
makes a face transparent when viewed from its front (normal toward camera) and
opaque when viewed from its back.

The PURE geometry (wall-quad tiling, arch parsing, outward-normal winding) lives
in render.py so it is unit-testable without bpy. This module only does
the bpy mesh + material assembly.
"""

NEUTRAL_RGB = (0.72, 0.72, 0.72)   # matte neutral shell material


def _get_or_create_material(bpy, name):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
    return m


def _node(nodes, type_name, label):
    """Reuse the default-named node if present, else create one (avoids strays)."""
    n = nodes.get(label)
    if n is None:
        n = nodes.new(type_name)
        n.name = label
    return n


def _floor_material(bpy, name="hsm_floor_mat"):
    """Plain opaque neutral material for the floor (NO backface culling — the
    floor must stay opaque when viewed from above)."""
    m = _get_or_create_material(bpy, name)
    nodes = m.node_tree.nodes
    links = m.node_tree.links
    links.clear()
    bsdf = _node(nodes, "ShaderNodeBsdfPrincipled", "Principled BSDF")
    out = _node(nodes, "ShaderNodeOutputMaterial", "Material Output")
    bsdf.inputs["Base Color"].default_value = (*NEUTRAL_RGB, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.9
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.2
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.2
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def _wall_backface_material(bpy, name="hsm_wall_bfc_mat"):
    """Neutral Principled BSDF mixed with transparent by the `Backfacing` geometry
    node: front faces (normal toward camera) -> transparent; back faces -> opaque.
    Mirrors blender_bpa.Builder.add_material(backface_culling=True) but shared."""
    m = _get_or_create_material(bpy, name)
    nodes = m.node_tree.nodes
    links = m.node_tree.links
    links.clear()
    bsdf = _node(nodes, "ShaderNodeBsdfPrincipled", "Principled BSDF")
    out = _node(nodes, "ShaderNodeOutputMaterial", "Material Output")
    bsdf.inputs["Base Color"].default_value = (*NEUTRAL_RGB, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.9
    mix = _node(nodes, "ShaderNodeMixShader", "Mix Shader")
    trans = _node(nodes, "ShaderNodeBsdfTransparent", "Transparent BSDF")
    geom = _node(nodes, "ShaderNodeNewGeometry", "Geometry")
    # Fac=Backfacing: 0 (front view) -> input[1]=transparent; 1 (back view) -> input[2]=bsdf
    links.new(geom.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(trans.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def _add_mesh(bpy, name, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata([tuple(v) for v in verts], [], [list(f) for f in faces])
    mesh.update()
    obj.data.materials.append(material)
    return obj


def build_shell(bpy, spec: dict):
    """Build floor + walls from a ShellSpec dict (see render.parse_shell_spec).

    Floor -> plain opaque neutral material. Walls -> flat quads with door/window
    openings cut out, carrying the backface-culling material (near walls
    transparent per-normal, far walls + their openings visible).
    """
    import render as rnd  # PURE geometry helpers (no bpy at module top)

    floor_verts = rnd.shell_floor_verts3d(spec)
    floor_mat = _floor_material(bpy)
    if len(floor_verts) >= 3:
        _add_mesh(bpy, "hsm_floor", floor_verts, [list(range(len(floor_verts)))], floor_mat)

    wall_mat = _wall_backface_material(bpy)
    wall_quads = rnd.shell_wall_quads(spec)
    if wall_quads:
        # One wall mesh object holding every wall quad — keeps the scene tidy.
        all_verts: list = []
        faces: list = []
        for q in wall_quads:
            base = len(all_verts)
            all_verts.extend(q)
            faces.append([base, base + 1, base + 2, base + 3])
        _add_mesh(bpy, "hsm_walls", all_verts, faces, wall_mat)
