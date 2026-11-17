bl_info = {
    "name": "Dovetail Key RU (Шпонка)",
    "author": "Milad Kambari / 3DRedbox + RU fork",
    "version": (5, 3, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > Шпонка",
    "description": "Этап 1: нарисовать линию и разрезать модель. Этап 2: шпонка/штифт + гнездо. Русский интерфейс, доп. формы ключей.",
    "category": "Object",
}

# Standalone version tuple. When installed as a 4.2+ extension there is no
# bl_info in the module namespace (extensions use blender_manifest.toml), so
# the panel must read this constant instead of bl_info.
MK_VERSION = (5, 3, 0)

# How much the tab's top ring is pulled inward relative to its height, giving
# the wall its conical (chamfered) slope. Shared by the operator and the
# preview so the guide always matches the result.
SEAM_TAPER = 0.6

# How close the pen has to come to its own starting point before the loop is
# pulled shut, in screen pixels. Generous on purpose: a loop that stops a few
# pixels short reads as a gap, and the user cannot tell whether it closed.
SNAP_PX = 34.0

import bpy
import bmesh
import gpu
import math
import heapq
import colorsys
import numpy as np
from math import radians
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from bpy_extras import view3d_utils

SMOOTH_ANGLE = 30.0  # degrees: below this stays smooth, above stays sharp

# dovetail default proportions
DT_WIDTH = 0.63
DT_TAPER = 0.6
DT_HEIGHT = 0.926
DT_DEPTH = 0.84



# ----------------------------------------------------------------------------
#  STAGE 1 : LINE SLICE  (do not change behaviour)
# ----------------------------------------------------------------------------

def _component(seed_face):
    seen = {seed_face}
    stack = [seed_face]
    while stack:
        f = stack.pop()
        for e in f.edges:
            for lf in e.link_faces:
                if lf not in seen:
                    seen.add(lf)
                    stack.append(lf)
    return seen


def _extract(faces):
    nb = bmesh.new()
    vmap = {}
    for f in faces:
        for v in f.verts:
            if v not in vmap:
                vmap[v] = nb.verts.new(v.co)
    nb.verts.index_update()
    for f in faces:
        try:
            nf = nb.faces.new([vmap[v] for v in f.verts])
            nf.material_index = f.material_index
        except ValueError:
            pass
    return nb


def _cap_and_clean(bm):
    """Watertight-cap all open boundaries. holes_fill first (clean ngon for
    planar cuts); whatever it can't close is capped with triangle_fill and
    coplanar triangles are merged back into ngons."""
    # Bisecting drops a new vertex wherever the plane meets an edge, and where
    # it passes a hair from an existing one that leaves a needle-thin face.
    # Those have no meaningful normal, which is exactly the crease that shows
    # up in the shading along a cut. Collapse them before capping.
    try:
        diag = _bm_diag(bm)
        eps = max(diag * 2e-5, 1e-9)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=eps)
        bmesh.ops.dissolve_degenerate(bm, dist=eps, edges=bm.edges[:])
    except Exception as ex:
        print("MK sliver cleanup error:", ex)

    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    if boundary:
        try:
            bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
        except Exception:
            pass
        remaining = [e for e in bm.edges
                     if e.is_valid and len(e.link_faces) == 1]
        if remaining:
            try:
                res = bmesh.ops.triangle_fill(
                    bm, use_beauty=True, use_dissolve=False, edges=remaining)
                newf = [f for f in res.get('geom', [])
                        if isinstance(f, bmesh.types.BMFace)]
                if newf:
                    inner = list({e for f in newf if f.is_valid
                                  for e in f.edges
                                  if len(e.link_faces) == 2
                                  and all(lf in newf for lf in e.link_faces)})
                    iverts = list({v for e in inner for v in e.verts})
                    if inner:
                        try:
                            bmesh.ops.dissolve_limit(
                                bm, angle_limit=0.0349,
                                verts=iverts, edges=inner)
                        except Exception:
                            pass
            except Exception:
                pass
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])


def smooth_keep_sharp(context, obj, angle_deg=SMOOTH_ANGLE):
    me = obj.data
    n = len(me.polygons)
    if n:
        me.polygons.foreach_set('use_smooth', [True] * n)
    me.update()
    ang = radians(angle_deg)

    if bpy.app.version < (4, 1, 0):
        try:
            me.use_auto_smooth = True
            me.auto_smooth_angle = ang
        except Exception:
            pass
        return

    prev = context.view_layer.objects.active
    prev_sel = obj.select_get()
    try:
        context.view_layer.objects.active = obj
        obj.select_set(True)
        done = False
        for kwargs in (dict(angle=ang), dict()):
            try:
                bpy.ops.object.shade_auto_smooth(**kwargs)
                done = True
                break
            except Exception:
                continue
        if not done:
            bm = bmesh.new()
            bm.from_mesh(me)
            for e in bm.edges:
                if len(e.link_faces) == 2 and e.calc_face_angle() > ang:
                    e.smooth = False
            bm.to_mesh(me)
            bm.free()
    finally:
        obj.select_set(prev_sel)
        context.view_layer.objects.active = prev


def _evaluated_mesh_copy(context, obj):
    """WYSIWYG: a copy of the mesh exactly as displayed (armature pose,
    subdivision and all other visible modifiers applied), in local space."""
    dg = context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    try:
        me = bpy.data.meshes.new_from_object(
            ev, preserve_all_data_layers=True, depsgraph=dg)
    except TypeError:
        me = bpy.data.meshes.new_from_object(ev)
    if not me.materials:
        for m in obj.data.materials:
            me.materials.append(m)
    return me


def _bm_diag(bm):
    if not bm.verts:
        return 1.0
    xs = [v.co[0] for v in bm.verts]
    ys = [v.co[1] for v in bm.verts]
    zs = [v.co[2] for v in bm.verts]
    return Vector((max(xs) - min(xs), max(ys) - min(ys),
                   max(zs) - min(zs))).length


def _commit_cut(context, obj, bm, comp, src_me):
    """Split bm into (rest, comp) and commit onto obj + a new _B object.
    Called only when the cut is known to have succeeded. Bakes the
    evaluated mesh into obj (modifiers cleared) - backup keeps the rig."""
    backup = _create_backup(context, obj)

    nb = _extract(comp)
    _cap_and_clean(nb)
    mesh_b = bpy.data.meshes.new(obj.data.name + "_B")
    for m in src_me.materials:
        mesh_b.materials.append(m)
    nb.to_mesh(mesh_b)
    nb.free()

    bmesh.ops.delete(bm, geom=list(comp), context='FACES')
    _cap_and_clean(bm)
    bm.to_mesh(src_me)
    bm.free()

    obj.modifiers.clear()
    obj.data = src_me
    obj.data.update()

    obj_b = bpy.data.objects.new(obj.name + "_B", mesh_b)
    obj_b.matrix_world = obj.matrix_world.copy()
    context.collection.objects.link(obj_b)
    if backup is not None:
        backup["mk_child_b"] = obj_b.name

    smooth_keep_sharp(context, obj)
    smooth_keep_sharp(context, obj_b)

    for o in list(context.selected_objects):
        o.select_set(False)
    obj.select_set(True)
    obj_b.select_set(True)
    context.view_layer.objects.active = obj
    return True


def do_slice(context, obj, a, b):
    src_me = None
    try:
        region = context.region
        rv3d = context.region_data

        o1 = view3d_utils.region_2d_to_origin_3d(region, rv3d, a)
        d1 = view3d_utils.region_2d_to_vector_3d(region, rv3d, a)
        o2 = view3d_utils.region_2d_to_origin_3d(region, rv3d, b)
        d2 = view3d_utils.region_2d_to_vector_3d(region, rv3d, b)

        normal_w = d1.cross((o2 + d2) - o1)
        if normal_w.length < 1e-9:
            return False
        normal_w.normalize()

        nA_w = d1.cross(normal_w)
        if (o2 + d2 - o1).dot(nA_w) < 0:
            nA_w = -nA_w
        nB_w = d2.cross(normal_w)
        if (o1 + d1 - o2).dot(nB_w) < 0:
            nB_w = -nB_w

        mw = obj.matrix_world
        mwi = mw.inverted()
        m3t = mw.to_3x3().transposed()

        co_l = mwi @ o1
        no_l = (m3t @ normal_w).normalized()
        oA_l = mwi @ o1
        nA_l = (m3t @ nA_w).normalized()
        oB_l = mwi @ o2
        nB_l = (m3t @ nB_w).normalized()

        src_me = _evaluated_mesh_copy(context, obj)
        bm = bmesh.new()
        bm.from_mesh(src_me)

        diag = _bm_diag(bm)
        eps = -1e-4 * max(diag, 1e-6)

        def in_slab(p):
            return ((p - oA_l).dot(nA_l) >= eps) and ((p - oB_l).dot(nB_l) >= eps)

        slab_faces = [f for f in bm.faces if in_slab(f.calc_center_median())]
        if not slab_faces:
            bm.free()
            bpy.data.meshes.remove(src_me)
            return False

        gverts, gedges = set(), set()
        for f in slab_faces:
            gverts.update(f.verts)
            gedges.update(f.edges)
        geom = list(gverts) + list(gedges) + list(slab_faces)

        res = bmesh.ops.bisect_plane(
            bm, geom=geom, dist=1e-5 * max(diag, 1e-6),
            plane_co=co_l, plane_no=no_l,
            clear_inner=False, clear_outer=False,
        )
        cut_edges = [e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
        if not cut_edges:
            bm.free()
            bpy.data.meshes.remove(src_me)
            return False

        bmesh.ops.split_edges(bm, edges=cut_edges)

        seed = None
        best = None
        for f in bm.faces:
            c = f.calc_center_median()
            if not in_slab(c):
                continue
            s = (c - co_l).dot(no_l)
            if s > 0 and (best is None or s > best):
                best = s
                seed = f
        if seed is None:
            bm.free()
            bpy.data.meshes.remove(src_me)
            return False

        comp = _component(seed)
        if len(comp) >= len(bm.faces):
            bm.free()
            bpy.data.meshes.remove(src_me)
            return False

        return _commit_cut(context, obj, bm, comp, src_me)
    except Exception as ex:
        print("MK Line Slice error:", ex)
        try:
            if src_me is not None and src_me.users == 0:
                bpy.data.meshes.remove(src_me)
        except Exception:
            pass
        return False


def _decimate_stroke(pts, max_pts=24, min_dist=8.0):
    out = [pts[0]]
    for p in pts[1:]:
        if (Vector(p) - Vector(out[-1])).length >= min_dist:
            out.append(p)
    if tuple(out[-1]) != tuple(pts[-1]):
        out.append(pts[-1])
    while len(out) > max_pts:
        dec = out[::2]
        if tuple(dec[-1]) != tuple(out[-1]):
            dec.append(out[-1])
        out = dec
    return out


def _smooth_stroke(pts, passes=4, chaikin=2, max_pts=32):
    """Remove hand jitter: gaussian smoothing + Chaikin corner cutting +
    uniform arc-length resample. Endpoints stay fixed."""
    P = [Vector(p) for p in _decimate_stroke(pts, max_pts=64, min_dist=3.0)]
    for _ in range(passes):
        if len(P) < 3:
            break
        Q = [P[0]]
        for i in range(1, len(P) - 1):
            Q.append((P[i - 1] + P[i] * 2.0 + P[i + 1]) / 4.0)
        Q.append(P[-1])
        P = Q
    for _ in range(chaikin):
        if len(P) < 3:
            break
        Q = [P[0]]
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            Q.append(a * 0.75 + b * 0.25)
            Q.append(a * 0.25 + b * 0.75)
        Q.append(P[-1])
        P = Q
    if len(P) > max_pts:
        d = [0.0]
        for i in range(1, len(P)):
            d.append(d[-1] + (P[i] - P[i - 1]).length)
        total = d[-1] or 1.0
        out = []
        ti = 0
        for k in range(max_pts):
            t = total * k / (max_pts - 1)
            while ti < len(d) - 2 and d[ti + 1] < t:
                ti += 1
            seg = (d[ti + 1] - d[ti]) or 1.0
            f = (t - d[ti]) / seg
            out.append(P[ti] * (1.0 - f) + P[ti + 1] * f)
        P = out
    return [(p.x, p.y) for p in P]


def _snap_verts_to_surface(obj, surf, snap_tol):
    """Pull every vertex within snap_tol of the surface exactly onto it, so
    the blade-thickness offset between the two pieces collapses to zero."""
    try:
        mw = obj.matrix_world
        mwi = mw.inverted()
        smw = surf.matrix_world
        smwi = smw.inverted()
        me = obj.data
        moved = 0
        for v in me.vertices:
            p_s = smwi @ (mw @ v.co)
            hit, loc, _nor, _idx = surf.closest_point_on_mesh(p_s)
            if hit and (p_s - loc).length <= snap_tol:
                v.co = mwi @ (smw @ loc)
                moved += 1
        me.update()
    except Exception as ex:
        print("MK snap error:", ex)


def _weld_caps(obj_a, obj_b, weld_tol, splane_co, snorm):
    """Zip the two cut rims together: only vertices lying within a thin band
    around the cut plane are considered (numpy prefilter), so the KD-tree
    stays tiny even on million-poly meshes. Paired verts move to the midpoint;
    the model form is untouched."""
    try:
        import numpy as np
        from mathutils.kdtree import KDTree
        ma = obj_a.matrix_world
        mb = obj_b.matrix_world
        mai = ma.inverted()
        mbi = mb.inverted()

        pc = np.array(splane_co)
        pn = np.array(snorm)
        band = weld_tol * 4.0

        def cap_indices(obj, M):
            me = obj.data
            n = len(me.vertices)
            co = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get('co', co)
            co = co.reshape(n, 3)
            world = co @ np.array(M.to_3x3()).T + np.array(M.translation)
            d = np.abs((world - pc) @ pn)
            idx = np.nonzero(d <= band)[0]
            return idx, world

        ia, wa = cap_indices(obj_a, ma)
        ib, wb = cap_indices(obj_b, mb)
        if len(ia) == 0 or len(ib) == 0:
            return

        kd = KDTree(len(ib))
        for k, j in enumerate(ib):
            kd.insert(Vector(wb[j]), k)
        kd.balance()

        va = obj_a.data.vertices
        vb = obj_b.data.vertices
        used = set()
        welded = 0
        for a_i in ia:
            wco = Vector(wa[a_i])
            loc, k, dist = kd.find(wco)
            if dist is not None and dist <= weld_tol and k not in used:
                b_i = int(ib[k])
                mid = (wco + loc) * 0.5
                va[a_i].co = mai @ mid
                vb[b_i].co = mbi @ mid
                used.add(k)
                welded += 1
        obj_a.data.update()
        obj_b.data.update()
        print("MK weld: %d vert pairs zipped (from %d/%d cap verts)"
              % (welded, len(ia), len(ib)))
    except Exception as ex:
        print("MK weld error:", ex)


def _blade_from_surface(context, surf, thickness):
    """Build a thin closed 'blade' solid from a cutting surface, for boolean
    difference. Returned as a linked object the caller must remove."""
    blade_src = surf.copy()
    blade_src.data = surf.data.copy()
    context.collection.objects.link(blade_src)
    sol = blade_src.modifiers.new("Solid", 'SOLIDIFY')
    sol.thickness = thickness
    sol.offset = 0.0
    blade_me = _evaluated_mesh_copy(context, blade_src)
    blade = bpy.data.objects.new("MK_Blade", blade_me)
    blade.matrix_world = blade_src.matrix_world.copy()
    context.collection.objects.link(blade)
    bpy.data.objects.remove(blade_src, do_unlink=True)
    return blade


def build_stroke_surface(context, obj, pts):
    """Build the ribbon surface the cut uses. The stroke is smoothed into a
    clean bezier-like curve first (hand jitter removed), so the cut is soft
    and free of the raw wobble."""
    try:
        if len(pts) < 2:
            return None
        region = context.region
        rv3d = context.region_data
        pts = _smooth_stroke(pts, passes=6, chaikin=3, max_pts=48)
        if len(pts) < 2:
            return None

        dg = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        mw = obj.matrix_world
        bbw = [mw @ Vector(c) for c in ev.bound_box]
        center = sum(bbw, Vector()) / 8.0

        vd = view3d_utils.region_2d_to_vector_3d(
            region, rv3d, (region.width / 2.0, region.height / 2.0)).normalized()
        proj = [(v - center).dot(vd) for v in bbw]
        depth_half = (max(proj) - min(proj)) * 0.5 + 1e-3
        mn_b = [min(v[i] for v in bbw) for i in range(3)]
        mx_b = [max(v[i] for v in bbw) for i in range(3)]
        pad = max((mx_b[i] - mn_b[i]) for i in range(3)) * 0.02 + 1e-4

        persp = bool(getattr(rv3d, "is_perspective", True))
        near_v, far_v = [], []
        for p in pts:
            o = view3d_utils.region_2d_to_origin_3d(region, rv3d, p)
            d = view3d_utils.region_2d_to_vector_3d(region, rv3d, p)
            tc = (center - o).dot(d)
            t0 = tc - depth_half - pad
            t1 = tc + depth_half + pad
            if persp:
                t0 = max(t0, 0.05)
                if t1 <= t0:
                    t1 = t0 + depth_half
            near_v.append(o + d * t0)
            far_v.append(o + d * t1)

        bmc = bmesh.new()
        vN = [bmc.verts.new(v) for v in near_v]
        vF = [bmc.verts.new(v) for v in far_v]
        bmc.verts.index_update()
        for i in range(len(pts) - 1):
            try:
                bmc.faces.new((vN[i], vN[i + 1], vF[i + 1], vF[i]))
            except ValueError:
                pass
        bmesh.ops.recalc_face_normals(bmc, faces=bmc.faces[:])
        me = bpy.data.meshes.new("MK_Surface")
        bmc.to_mesh(me)
        bmc.free()
        if len(me.polygons):
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
            me.update()
        surf = bpy.data.objects.new("MK_Surface", me)
        surf.show_wire = True
        context.collection.objects.link(surf)
        return surf
    except Exception as ex:
        print("MK surface error:", ex)
        return None


def do_surface_cut(context, obj, surf):
    """Cut obj into two pieces using an existing MK_Surface ribbon. Same
    reliable blade-boolean path as freehand, but the surface is supplied by
    the user (view-independent)."""
    try:
        if surf.type != 'MESH' or len(surf.data.polygons) == 0:
            return False

        bb = [Vector(c) for c in obj.bound_box]
        diagw = (Vector((max(v[i] for v in bb) for i in range(3)))
                 - Vector((min(v[i] for v in bb) for i in range(3)))).length

        blade_thickness = 0.00002
        src_me = _evaluated_mesh_copy(context, obj)
        # keep a pristine copy so each solver attempt starts from the same
        # un-boolean'd geometry (the first attempt mutates obj.data)
        pristine = src_me.copy()
        backup = _create_backup(context, obj)
        obj.modifiers.clear()
        obj.data = src_me
        obj.data.update()

        # Bake the cut. A hair-thin blade often fails to fully divide a dense
        # or slightly non-manifold character, and the split is only accepted
        # when the mesh really came apart, so try each solver and then step the
        # blade thicker until it does. _weld_caps zips the two rims back
        # together afterwards, so a thicker blade does not show.
        # Solver order on a dense mesh, measured on a million-poly body:
        #   MANIFOLD  0.3 s  but never actually separated it
        #   FAST      2.6 s  separated it
        #   EXACT     9.1 s  separated it
        # So FAST first: it is the cheapest one that does the job. Putting
        # MANIFOLD first only spends time on attempts that cannot succeed, and
        # leaving EXACT first made every heavy cut wait three times longer
        # than it needed to.
        heavy = len(obj.data.polygons) > 250000
        order = ('FAST', 'EXACT', 'MANIFOLD') if heavy \
            else ('EXACT', 'FAST', 'MANIFOLD')
        attempts = []
        muls = (8.0, 40.0, 200.0) if heavy else (1.0, 8.0, 40.0, 200.0)
        for mul in muls:
            for sv in order:
                attempts.append((blade_thickness * mul, sv))
        split_ok = False
        saved = None
        used_thickness = blade_thickness
        for si, (thick, solver) in enumerate(attempts):
            if si > 0:
                # Free the mesh from the failed attempt before making the next
                # copy. Without this every retry left a full-size mesh behind
                # with nobody using it - twelve attempts on a million-poly
                # model piles up gigabytes and takes Blender down with it.
                stale = obj.data
                obj.data = pristine.copy()
                obj.data.update()
                try:
                    if stale.users == 0:
                        bpy.data.meshes.remove(stale)
                except Exception:
                    pass
            used_thickness = thick
            blade2 = _blade_from_surface(context, surf, thick)
            ok = _apply_boolean(context, obj, blade2, 'DIFFERENCE', solver)
            bpy.data.objects.remove(blade2, do_unlink=True)
            if not ok or len(obj.data.polygons) == 0:
                continue
            smw = surf.matrix_world
            sverts = [smw @ v.co for v in surf.data.vertices]
            splane_co = sum(sverts, Vector()) / max(len(sverts), 1)
            snorm = Vector((0.0, 0.0, 0.0))
            for poly in surf.data.polygons:
                snorm += (smw.to_3x3() @ poly.normal)
            if snorm.length < 1e-9:
                snorm = Vector((0.0, 0.0, 1.0))
            snorm.normalize()
            # The ribbon is an open sheet, so its winding - and therefore this
            # normal - flips when the stroke is drawn right-to-left instead of
            # left-to-right. That swapped which piece kept the original name
            # and stayed active, so the seam looked like it landed on the wrong
            # half (or nowhere). Pin the sign to the dominant axis: the split
            # is now the same whichever way the line is drawn. Geometry is
            # untouched, only which side is called A.
            _ax = max(range(3), key=lambda i: abs(snorm[i]))
            if snorm[_ax] < 0.0:
                snorm = -snorm
            omw = obj.matrix_world
            bm = bmesh.new()
            bm.from_mesh(obj.data)

            # gather all loose shells
            shells = []
            seen = set()
            for f0 in bm.faces:
                if f0 in seen:
                    continue
                compf = list(_component(f0))
                seen.update(compf)
                cen = Vector((0.0, 0.0, 0.0))
                for f in compf:
                    cen += f.calc_center_median()
                cen /= len(compf)
                shells.append({"faces": compf, "cen_w": omw @ cen,
                               "n": len(compf)})

            # 2.0.1 rule: classify each SHELL as a whole by its centroid. A
            # shell is never broken up, so the infinite cut plane cannot sever
            # a limb the blade never reached. If the blade did not actually
            # separate the piece, one side comes out empty and the next solver
            # is tried; when none of them separate it, the cut is refused
            # rather than slicing the whole body along the plane.
            fa, fb = [], []
            for sh in shells:
                s = (sh["cen_w"] - splane_co).dot(snorm)
                (fa if s >= 0.0 else fb).extend(sh["faces"])

            if not (fa and fb) and len(shells) >= 2:
                # The boolean DID separate the mesh - the plane test just could
                # not tell the pieces apart, which happens whenever both hang
                # the same way (a ring taken round an arm: body and arm
                # centres are both below it). Trust the shells instead of
                # throwing a good cut away: biggest is the body, next is the
                # piece that came off, and any small loose parts go to
                # whichever half they sit nearest.
                from mathutils.kdtree import KDTree
                order = sorted(shells, key=lambda s: s["n"], reverse=True)
                fa = list(order[0]["faces"])
                fb = list(order[1]["faces"])
                rest = order[2:]
                if rest:
                    def _kd_of(faces):
                        vs = list({v for f in faces for v in f.verts})
                        kd = KDTree(len(vs))
                        for i, v in enumerate(vs):
                            kd.insert(v.co, i)
                        kd.balance()
                        return kd
                    kd_a = _kd_of(fa)
                    kd_b = _kd_of(fb)
                    for sh in rest:
                        fs = sh["faces"]
                        sample = fs[:: max(len(fs) // 12, 1)]
                        da = db = 1e18
                        for f in sample:
                            c = f.calc_center_median()
                            da = min(da, kd_a.find(c)[2])
                            db = min(db, kd_b.find(c)[2])
                        (fa if da <= db else fb).extend(fs)
            if fa and fb:
                split_ok = True
                saved = (bm, fa, fb)
                break
            bm.free()

        if not split_ok:
            try:
                if pristine.users == 0:
                    bpy.data.meshes.remove(pristine)
            except Exception:
                pass
            _undo_last_cut(context)
            return False

        bm, fa, fb = saved
        try:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)
        except Exception:
            pass

        nb = _extract(fb)
        try:
            d2 = _bm_diag(nb)
            e2 = max(d2 * 2e-5, 1e-9)
            bmesh.ops.remove_doubles(nb, verts=nb.verts[:], dist=e2)
            bmesh.ops.dissolve_degenerate(nb, dist=e2, edges=nb.edges[:])
        except Exception:
            pass
        mesh_b = bpy.data.meshes.new(obj.data.name + "_B")
        for m in obj.data.materials:
            mesh_b.materials.append(m)
        nb.to_mesh(mesh_b)
        nb.free()
        na = _extract(fa)
        try:
            d3 = _bm_diag(na)
            e3 = max(d3 * 2e-5, 1e-9)
            bmesh.ops.remove_doubles(na, verts=na.verts[:], dist=e3)
            bmesh.ops.dissolve_degenerate(na, dist=e3, edges=na.edges[:])
        except Exception:
            pass
        na.to_mesh(obj.data)
        na.free()
        bm.free()
        obj.data.update()

        obj_b = bpy.data.objects.new(obj.name + "_B", mesh_b)
        obj_b.matrix_world = obj.matrix_world.copy()
        context.collection.objects.link(obj_b)
        if backup is not None:
            backup["mk_child_b"] = obj_b.name

        # zip the two rims together so the pieces mate flush (form untouched)
        _weld_caps(obj, obj_b, used_thickness * 6.0, splane_co, snorm)

        smooth_keep_sharp(context, obj)
        smooth_keep_sharp(context, obj_b)

        for o in list(context.selected_objects):
            o.select_set(False)
        obj.select_set(True)
        obj_b.select_set(True)
        context.view_layer.objects.active = obj
        return True
    except Exception as ex:
        print("MK surface-cut error:", ex)
        import traceback
        traceback.print_exc()
        return False


# ----------------------------------------------------------------------------
#  STAGE 2 : CUBE AT CROSS-SECTION CENTER  (new, independent button)
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
#  STAGE 2 : CUBE AT CROSS-SECTION CENTER  (new, independent button)
# ----------------------------------------------------------------------------

def find_section(obj, want_faces=False):
    """Largest flat (coplanar, adjacency-connected) face cluster = the cut
    cross-section. Returns (center_world, normal_world, radius_world), plus
    face_indices when want_faces=True."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    bm.faces.ensure_lookup_table()

    cos_tol = 0.999  # ~2.5 deg
    visited = set()
    best = None  # (area, cluster, normal)
    for f0 in bm.faces:
        if f0 in visited or f0.normal.length < 1e-9:
            continue
        n0 = f0.normal.copy()
        cluster = []
        stack = [f0]
        visited.add(f0)
        while stack:
            f = stack.pop()
            cluster.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf in visited or lf.normal.length < 1e-9:
                        continue
                    if lf.normal.dot(n0) >= cos_tol:
                        visited.add(lf)
                        stack.append(lf)
        area = sum(f.calc_area() for f in cluster)
        if best is None or area > best[0]:
            best = (area, cluster, n0)

    if best is None:
        bm.free()
        return None

    _, cluster, n0 = best
    face_idx = [f.index for f in cluster]
    cverts = set()
    tot = 0.0
    cen = Vector((0.0, 0.0, 0.0))
    for f in cluster:
        a = f.calc_area()
        cen += f.calc_center_median() * a
        tot += a
        cverts.update(f.verts)
    if tot <= 0.0:
        bm.free()
        return None
    cen /= tot

    mw = obj.matrix_world
    center_w = mw @ cen
    normal_w = (mw.to_3x3() @ n0).normalized()
    radius_w = 0.0
    for v in cverts:
        radius_w = max(radius_w, (mw @ v.co - center_w).length)
    bm.free()
    if radius_w <= 0.0:
        return None
    if want_faces:
        return center_w, normal_w, radius_w, face_idx
    return center_w, normal_w, radius_w


def _seam_cap_indices(obj, other):
    """Return the set of face indices on `obj` that form the shared cut cap
    with `other` (the faces sitting face-to-face across the seam). Same
    detection as find_section_mutual, but returns the actual faces so the
    groove operates on EXACTLY the cap - not a guessed plane band."""
    try:
        import numpy as np
        from mathutils.bvhtree import BVHTree
        mw = obj.matrix_world
        omw = other.matrix_world
        omwi = omw.inverted()
        bb = [Vector(c) for c in obj.bound_box]
        diag = (Vector((max(v[i] for v in bb) for i in range(3)))
                - Vector((min(v[i] for v in bb) for i in range(3)))).length
        me = obj.data
        n_f = len(me.polygons)
        if n_f == 0:
            return set()
        centers = np.empty(n_f * 3, dtype=np.float64)
        me.polygons.foreach_get('center', centers)
        centers = centers.reshape(n_f, 3)
        M = np.array(mw.to_3x3()); t = np.array(mw.translation)
        cw = centers @ M.T + t
        obb = [omw @ Vector(c) for c in other.bound_box]
        lo = np.min(np.array(obb), axis=0)
        hi = np.max(np.array(obb), axis=0)
        dg = bpy.context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(other, dg)
        cand = set()
        for tol_f in (8e-4, 3e-3, 1e-2):
            tol = max(tol_f * diag, 1e-6)
            box = ((cw >= (lo - tol)) & (cw <= (hi + tol))).all(axis=1)
            idx = np.nonzero(box)[0]
            cand = set()
            for j in idx:
                p_o = omwi @ Vector(cw[j])
                loc, _n, _fi, dist = bvh.find_nearest(p_o)
                if loc is not None and dist <= tol:
                    cand.add(int(j))
            if cand:
                break
        if not cand:
            return set()
        # keep only the largest CONNECTED cluster so scattered stray faces
        # (near the other half's bbox but not on the cut cap) are excluded
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        left = set(cand)
        best = None
        while left:
            f0 = left.pop()
            cluster = {f0}
            stack = [f0]
            while stack:
                fi = stack.pop()
                for e in bm.faces[fi].edges:
                    for lf in e.link_faces:
                        li = lf.index
                        if li in left:
                            left.discard(li)
                            cluster.add(li)
                            stack.append(li)
            if best is None or len(cluster) > len(best):
                best = cluster
        bm.free()
        return best or cand
    except Exception as ex:
        print("MK seam cap error:", ex)
        return set()


def find_section_mutual(obj, other):
    """Section info from the two halves: cap = largest connected cluster of
    obj's faces that sit face-to-face with the other half. AABB prefilter +
    BVH lookups keep it fast on million-poly pieces."""
    bm = None
    try:
        import numpy as np
        from mathutils.bvhtree import BVHTree

        mw = obj.matrix_world
        n3 = mw.to_3x3()
        omw = other.matrix_world
        omwi = omw.inverted()

        bb = [Vector(c) for c in obj.bound_box]
        diag = (Vector((max(v[i] for v in bb) for i in range(3)))
                - Vector((min(v[i] for v in bb) for i in range(3)))).length

        me = obj.data
        n_f = len(me.polygons)
        centers = np.empty(n_f * 3, dtype=np.float64)
        me.polygons.foreach_get('center', centers)
        centers = centers.reshape(n_f, 3)
        # to world
        M = np.array(mw.to_3x3())
        t = np.array(mw.translation)
        cw = centers @ M.T + t

        # AABB of the other piece (world), expanded
        obb = [omw @ Vector(c) for c in other.bound_box]
        lo = np.min(np.array(obb), axis=0)
        hi = np.max(np.array(obb), axis=0)

        dg = bpy.context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(other, dg)

        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()

        cand = set()
        for tol_f in (8e-4, 3e-3, 1e-2):
            tol = max(tol_f * diag, 1e-6)
            box = ((cw >= (lo - tol)) & (cw <= (hi + tol))).all(axis=1)
            idx = np.nonzero(box)[0]
            cand = set()
            for j in idx:
                p_o = omwi @ Vector(cw[j])
                loc, _nor, _fi, dist = bvh.find_nearest(p_o)
                if loc is not None and dist <= tol:
                    cand.add(bm.faces[int(j)])
            if cand:
                break
        if not cand:
            bm.free()
            return None

        best = None
        left = set(cand)
        while left:
            f0 = left.pop()
            cluster = {f0}
            stack = [f0]
            while stack:
                f = stack.pop()
                for e in f.edges:
                    for lf in e.link_faces:
                        if lf in left:
                            left.discard(lf)
                            cluster.add(lf)
                            stack.append(lf)
            area = sum(f.calc_area() for f in cluster)
            if best is None or area > best[0]:
                best = (area, cluster)
        _area, cap = best

        tot = 0.0
        cen_w = Vector((0.0, 0.0, 0.0))
        nrm_w = Vector((0.0, 0.0, 0.0))
        cap_verts = set()
        for f in cap:
            a = f.calc_area()
            tot += a
            cen_w += (mw @ f.calc_center_median()) * a
            nrm_w += (n3 @ f.normal) * a
            cap_verts.update(f.verts)
        if tot <= 0.0 or nrm_w.length < 1e-9:
            bm.free()
            return None
        cen_w /= tot
        normal_w = nrm_w.normalized()
        # robust radius: median vertex distance (not max), vectorised so it
        # stays fast even when the cap has many verts
        cap_list = list(cap_verts)
        cco = np.array([v.co[:] for v in cap_list])
        Mw = np.array(mw.to_3x3())
        tw = np.array(mw.translation)
        cw2 = cco @ Mw.T + tw
        dists = np.linalg.norm(cw2 - np.array(cen_w), axis=1)
        if len(dists) == 0:
            bm.free()
            return None
        radius_w = float(np.median(dists)) * 1.4
        if radius_w <= 1e-9:
            radius_w = float(dists.max())
        print("MK dovetail: mutual cap faces=%d of %d candidates, "
              "center=%s radius=%.4f"
              % (len(cap), len(cand),
                 tuple(round(x, 3) for x in cen_w), radius_w))
        bm.free()
        if radius_w <= 0.0:
            return None
        return cen_w, normal_w, radius_w
    except Exception as ex:
        print("MK mutual-section error:", ex)
        try:
            if bm is not None:
                bm.free()
        except Exception:
            pass
        return None


def _mesh_from(name, verts, faces):
    bm = bmesh.new()
    bvs = [bm.verts.new(v) for v in verts]
    bm.verts.index_update()
    for f in faces:
        try:
            bm.faces.new([bvs[i] for i in f])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def _ball_peg_mesh(name, neck_r, ball_r, neck_len, ball_offset, segments=24):
    """Side A PEG: a ball-and-neck peg built as ONE continuous lathed surface
    (a profile revolved around +Z), so the neck and ball are a single
    watertight shell with no internal wall. Profile: flat base cap, straight
    neck up to where it meets the ball, then the ball's spherical arc up to the
    top. Built along local +Z; neck base at z=-neck_len, ball centre at
    z=ball_offset."""
    import math
    seg = max(segments, 12)
    r_ball = max(ball_r, 1e-5)
    r_neck = max(min(neck_r, r_ball * 0.98), 1e-6)
    z_base = -neck_len
    z_ball = ball_offset

    # angle on the ball where the neck meets it (neck radius = ball slice)
    # ball slice radius at angle a (from -Z pole): r_ball*sin(a); we want the
    # neck to join the lower hemisphere where the ball radius == r_neck
    a_join = math.asin(max(min(r_neck / r_ball, 1.0), 0.0))  # from bottom pole
    # z where they join (below ball centre)
    z_join = z_ball - r_ball * math.cos(a_join)

    # build the 2D profile (radius, z) from base to top
    prof = []
    prof.append((0.0, z_base))              # centre of base cap
    prof.append((r_neck, z_base))           # base rim
    prof.append((r_neck, z_join))           # up the neck to the ball join
    # ball arc from the join up and over the top
    arc_steps = max(seg // 2, 8)
    a_start = -math.cos(a_join)             # cos param from join (negative)
    # sweep polar angle from the join angle to the top pole
    ang0 = math.pi - a_join                 # measured from +Z top... use direct
    # simpler: parametrize by theta measured from bottom pole (0..pi)
    theta0 = a_join                         # at the join (near bottom)
    theta1 = math.pi                        # top pole
    for i in range(1, arc_steps + 1):
        t = theta0 + (theta1 - theta0) * (i / arc_steps)
        rad = r_ball * math.sin(t)
        zz = z_ball - r_ball * math.cos(t)
        prof.append((max(rad, 0.0), zz))
    # ensure last point is the exact top pole (radius 0)
    prof[-1] = (0.0, z_ball + r_ball)

    # revolve the profile around Z
    bm = bmesh.new()
    rings = []
    for (rad, zz) in prof:
        if rad <= 1e-9:
            rings.append([bm.verts.new((0.0, 0.0, zz))])   # pole (single vert)
        else:
            ring = []
            for k in range(seg):
                a = (2.0 * math.pi * k) / seg
                ring.append(bm.verts.new((rad * math.cos(a),
                                          rad * math.sin(a), zz)))
            rings.append(ring)
    bm.verts.index_update()

    for i in range(len(rings) - 1):
        A = rings[i]
        B = rings[i + 1]
        if len(A) == 1 and len(B) > 1:          # bottom pole fan
            for k in range(len(B)):
                j = (k + 1) % len(B)
                try:
                    bm.faces.new((A[0], B[j], B[k]))
                except ValueError:
                    pass
        elif len(B) == 1 and len(A) > 1:        # top pole fan
            for k in range(len(A)):
                j = (k + 1) % len(A)
                try:
                    bm.faces.new((A[k], A[j], B[0]))
                except ValueError:
                    pass
        elif len(A) > 1 and len(B) > 1:         # quad band
            for k in range(len(A)):
                j = (k + 1) % len(A)
                try:
                    bm.faces.new((A[k], A[j], B[j], B[k]))
                except ValueError:
                    pass

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def _ball_socket_mesh(name, neck_r, ball_r, neck_len, ball_offset, depth,
                      segments=24, clearance=0.0):
    """Side B SOCKET (capsule bore): a straight round hole the width of the
    BALL (plus clearance), opening at the cut plane and closing with a
    hemispherical dome. Entry is ball-wide so the peg drops straight in.

    clearance widens the bore radius uniformly (ball_r + ball_r*clearance) so
    the hole stays perfectly ROUND and concentric - never an ellipse.

    Local space: bored along +Z. Mouth at z=0 (cut plane); cylinder up to
    z=+cyl_len, then a hemisphere caps it.
    """
    import math
    # round bore, widened uniformly by the clearance offset
    r = ball_r * (1.0 + max(clearance, 0.0))
    # `depth` is the total reach of the bore past the cut plane (matched to
    # the peg + a small offset by the caller). The dome accounts for the last
    # r of that reach; the straight cylinder makes up the rest.
    bore_len = max(depth, r * 1.05)
    cyl_len = max(bore_len - r, r * 0.05)
    # start the mouth slightly OUTSIDE the cut plane (negative z) so the
    # opening always fully punches through the surface even when the cut
    # face is curved - otherwise a lip is left around the mouth.
    overshoot = max(r * 0.6, bore_len * 0.25)

    bm = bmesh.new()
    seg = max(segments, 16)
    rings = []  # (z, radius)

    rings.append((-overshoot, r))
    rings.append((0.0, r))
    rings.append((cyl_len, r))
    dome_steps = max(seg // 2, 8)
    for i in range(1, dome_steps + 1):
        t = i / dome_steps
        ang = t * (math.pi / 2.0)
        z = cyl_len + r * math.sin(ang)
        rad = r * math.cos(ang)
        rings.append((z, max(rad, 1e-5)))

    ring_verts = []
    for (z, rad) in rings:
        vs = []
        for k in range(seg):
            a = (2.0 * math.pi * k) / seg
            vs.append(bm.verts.new((rad * math.cos(a), rad * math.sin(a), z)))
        ring_verts.append(vs)
    bm.verts.index_update()

    for rr in range(len(ring_verts) - 1):
        A = ring_verts[rr]
        B = ring_verts[rr + 1]
        for k in range(seg):
            j = (k + 1) % seg
            try:
                bm.faces.new((A[k], A[j], B[j], B[k]))
            except ValueError:
                pass
    # mouth cap at the overshoot end (z=-overshoot)
    try:
        bm.faces.new(ring_verts[0])
    except ValueError:
        for k in range(1, seg - 1):
            try:
                bm.faces.new((ring_verts[0][0], ring_verts[0][k],
                              ring_verts[0][k + 1]))
            except ValueError:
                pass
    # dome tip
    tip = bm.verts.new((0.0, 0.0, cyl_len + r))
    last = ring_verts[-1]
    for k in range(seg):
        j = (k + 1) % seg
        try:
            bm.faces.new((last[j], last[k], tip))
        except ValueError:
            pass

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me




def _cylinder_pin_mesh(name, radius, height, segments=24):
    """Round anti-pull pin along local +Z, centered at origin."""
    import math
    bm = bmesh.new()
    seg = max(segments, 12)
    r = max(radius, 1e-6)
    hh = max(height, 1e-6) * 0.5
    bot = [bm.verts.new((r * math.cos(2 * math.pi * k / seg),
                         r * math.sin(2 * math.pi * k / seg), -hh))
           for k in range(seg)]
    top = [bm.verts.new((r * math.cos(2 * math.pi * k / seg),
                         r * math.sin(2 * math.pi * k / seg), hh))
           for k in range(seg)]
    bm.verts.index_update()
    try:
        bm.faces.new(list(reversed(bot)))
    except ValueError:
        pass
    try:
        bm.faces.new(top)
    except ValueError:
        pass
    for k in range(seg):
        j = (k + 1) % seg
        try:
            bm.faces.new((bot[k], bot[j], top[j], top[k]))
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def _box_pin_mesh(name, width, depth, height):
    """Square/rectangular anti-rotation pin along +Z."""
    wb, db, hh = max(width, 1e-6) * 0.5, max(depth, 1e-6) * 0.5, max(height, 1e-6) * 0.5
    verts = [
        (-wb, -db, -hh), (wb, -db, -hh), (wb, db, -hh), (-wb, db, -hh),
        (-wb, -db, hh), (wb, -db, hh), (wb, db, hh), (-wb, db, hh),
    ]
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    return _mesh_from(name, verts, faces)


def _hex_pin_mesh(name, radius, height, segments=6):
    """Hexagonal prism pin (anti-rotation) along +Z."""
    return _cylinder_pin_mesh(name, radius, height, segments=max(segments, 6))


def _tapered_pin_mesh(name, r_base, r_tip, height, segments=24):
    """Self-centering truncated cone along +Z."""
    import math
    bm = bmesh.new()
    seg = max(segments, 12)
    rb, rt = max(r_base, 1e-6), max(r_tip, 1e-6)
    hh = max(height, 1e-6) * 0.5
    bot = [bm.verts.new((rb * math.cos(2 * math.pi * k / seg),
                         rb * math.sin(2 * math.pi * k / seg), -hh))
           for k in range(seg)]
    top = [bm.verts.new((rt * math.cos(2 * math.pi * k / seg),
                         rt * math.sin(2 * math.pi * k / seg), hh))
           for k in range(seg)]
    bm.verts.index_update()
    try:
        bm.faces.new(list(reversed(bot)))
    except ValueError:
        pass
    try:
        bm.faces.new(top)
    except ValueError:
        pass
    for k in range(seg):
        j = (k + 1) % seg
        try:
            bm.faces.new((bot[k], bot[j], top[j], top[k]))
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def _cross_pin_mesh(name, arm_w, arm_len, height):
    """Plus/cross key — strong against rotation."""
    w = max(arm_w, 1e-6) * 0.5
    L = max(arm_len, 1e-6) * 0.5
    hh = max(height, 1e-6) * 0.5
    # two overlapping boxes merged via bmesh
    bm = bmesh.new()
    for sx, sy in ((L, w), (w, L)):
        coords = [
            (-sx, -sy, -hh), (sx, -sy, -hh), (sx, sy, -hh), (-sx, sy, -hh),
            (-sx, -sy, hh), (sx, -sy, hh), (sx, sy, hh), (-sx, sy, hh),
        ]
        vs = [bm.verts.new(c) for c in coords]
        bm.verts.index_update()
        for f in (
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ):
            try:
                bm.faces.new([vs[i] for i in f])
            except ValueError:
                pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def _tslot_pin_mesh(name, stem_w, stem_d, head_w, head_d, height):
    """T-tenon: stem + wider head — slides in one way, locks against pull."""
    sw, sd = max(stem_w, 1e-6) * 0.5, max(stem_d, 1e-6) * 0.5
    hw, hd = max(head_w, 1e-6) * 0.5, max(head_d, 1e-6) * 0.5
    hh = max(height, 1e-6) * 0.5
    # stem from z=-hh to 0, head from 0 to +hh
    verts = [
        # stem bottom
        (-sw, -sd, -hh), (sw, -sd, -hh), (sw, sd, -hh), (-sw, sd, -hh),
        # stem/head join at z=0
        (-sw, -sd, 0.0), (sw, -sd, 0.0), (sw, sd, 0.0), (-sw, sd, 0.0),
        # head outer at z=0
        (-hw, -hd, 0.0), (hw, -hd, 0.0), (hw, hd, 0.0), (-hw, hd, 0.0),
        # head top
        (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh),
    ]
    faces = [
        (0, 1, 2, 3),  # stem bottom
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # stem walls
        # flange ring at z=0 (outer head - inner stem) approx as head bottom+top
        (8, 9, 13, 12), (9, 10, 14, 13), (10, 11, 15, 14), (11, 8, 12, 15),
        (12, 13, 14, 15),  # head top
        # connect stem top rim to head inner — simplified solid: stem extruded into head block
        (4, 5, 9, 8), (5, 6, 10, 9), (6, 7, 11, 10), (7, 4, 8, 11),
    ]
    return _mesh_from(name, verts, faces)


def _prism_key_pair(name_m, name_f, verts, faces, clearance):
    """Male mesh + uniformly scaled female mesh (clearance)."""
    me_m = _mesh_from(name_m, verts, faces)
    me_f = _mesh_from(name_f, verts, faces)
    return me_m, me_f, 1.0 + max(clearance, 0.0)



def _key_offset_world(center_w, normal_w, radius_w, scene):
    """Return the world-space centre for the key, shifted across the cut face
    by the Move X / Move Y sliders. The shift happens IN the cut plane (perp to
    the section normal), so the key stays on the same interface, just moved to
    where the user wants it."""
    ox = getattr(scene, "mk_off_x", 0.0)
    oy = getattr(scene, "mk_off_y", 0.0)
    if abs(ox) < 1e-9 and abs(oy) < 1e-9:
        return center_w.copy()
    n = normal_w.normalized()
    up = Vector((0.0, 0.0, 1.0))
    if abs(n.dot(up)) > 0.95:
        up = Vector((0.0, 1.0, 0.0))
    u = n.cross(up).normalized()
    v = n.cross(u).normalized()
    return center_w + u * (ox * radius_w) + v * (oy * radius_w)


def _z_translation(z):
    from mathutils import Matrix
    return Matrix.Translation((0.0, 0.0, z))


class MK_OT_section_dovetail(bpy.types.Operator):
    """Add a dovetail peg to one half and the matching socket to the other.

Select both halves, then click.
If no key appears, turn on Manual and change Scale or Height, then retry"""
    bl_idname = "mk.section_dovetail"
    bl_label = "Шпонка"
    bl_options = {'REGISTER', 'UNDO'}

    key_shape: bpy.props.EnumProperty(
        name="Форма ключа",
        description="Профиль соединения: ласточкин хвост, шар, цилиндр и др.",
        items=[
            ('DOVETAIL', "Ласточкин хвост", "Классический клиновой шип"),
            ('BALL', "Шар", "Шар на шейке (как замочная скважина)"),
            ('CYLINDER', "Цилиндр", "Круглый штифт"),
            ('BOX', "Коробка", "Квадратный штифт против проворота"),
            ('HEX', "Шестигранник", "Шестигранный штифт"),
            ('TAPERED', "Конус", "Самоцентрирующийся усечённый конус"),
            ('CROSS', "Крест", "Крестообразный ключ против проворота"),
            ('TSLOT', "Т-шип", "Т-образный шип (скользит и держит)"),
        ],
        default='DOVETAIL',
    )
    ball_size: bpy.props.FloatProperty(
        name="Размер шара",
        description="Радиус шара как доля радиуса сечения",
        default=0.45, min=0.05, max=2.0,
    )
    neck_ratio: bpy.props.FloatProperty(
        name="Шейка",
        description="Радиус шейки как доля радиуса шара",
        default=0.55, min=0.1, max=0.95,
    )
    width: bpy.props.FloatProperty(
        name="Ширина",
        description="Ширина основания как доля радиуса сечения",
        default=DT_WIDTH, min=0.01, max=3.0,
    )
    taper: bpy.props.FloatProperty(
        name="Конусность",
        description="Отношение верха к основанию (угол хвоста). <1 — раструб",
        default=DT_TAPER, min=0.05, max=1.0,
    )
    height: bpy.props.FloatProperty(
        name="Высота",
        description="Высота шипа вдоль нормали сечения (доля радиуса)",
        default=DT_HEIGHT, min=0.05, max=5.0,
    )
    depth: bpy.props.FloatProperty(
        name="Глубина",
        description="Длина вдоль направления скольжения (доля радиуса)",
        default=DT_DEPTH, min=0.05, max=5.0,
    )
    clearance: bpy.props.FloatProperty(
        name="Зазор",
        description="Насколько гнездо больше ключа (доля)",
        default=0.03, min=0.0, max=0.5,
    )
    swap: bpy.props.BoolProperty(
        name="Поменять стороны",
        description="Поменять, какая половина получает шип и гнездо",
        default=False,
    )
    solver: bpy.props.EnumProperty(
        name="Солвер",
        items=[('FAST', "Быстрый", "Float/быстрый — стабильнее на тяжёлых сетках"),
               ('MANIFOLD', "Manifold", "Fast & clean for closed (manifold) meshes"),
               ('EXACT', "Exact", "Cleaner, slower, can choke on dense sculpts")],
        default='FAST',
    )

    def execute(self, context):
        sel = [o for o in context.selected_objects if o.type == 'MESH']
        halves = list(sel)
        if len(halves) != 2:
            self.report({'ERROR'},
                        "Выделите две половины после реза")
            return {'CANCELLED'}

        ref = context.active_object
        if ref not in halves:
            ref = halves[0]
        other = halves[0] if ref is halves[1] else halves[1]

        # 1) best: the shared interface between the two halves
        sec = find_section_mutual(ref, other)
        if sec is None:
            sec = find_section_mutual(other, ref)
        # 2) last resort: flat-section finder
        if sec is None:
            sec = find_section(ref)
        if sec is None:
            self.report({'ERROR'}, "Сечение не найдено")
            return {'CANCELLED'}
        center_w, normal_w, radius_w = sec

        scene = context.scene
        # slide the key across the cut face if the user set Move X / Move Y
        center_w = _key_offset_world(center_w, normal_w, radius_w, scene)

        if getattr(scene, "mk_manual", False):
            size_ref = max(radius_w * scene.mk_scale, 1e-5)
            h_factor = scene.mk_height
        else:
            size_ref = radius_w
            h_factor = self.height

        # socket oversize: use the panel slider (always available so the user
        # can size the hole in side B before generating)
        clearance = getattr(scene, "mk_clearance", self.clearance)

        sz = -1.0 if getattr(scene, "mk_flip", False) else 1.0

        shape = self.key_shape
        fscale = 1.0 + max(clearance, 0.0)
        if shape == 'BALL':
            ball_r = max(size_ref * self.ball_size, 1e-5)
            neck_r = max(ball_r * self.neck_ratio, 1e-6)
            neck_len = ball_r * 0.6
            ball_off = ball_r * (0.9 + h_factor * 1.6)
            peg_reach = ball_off + ball_r
            socket_depth = peg_reach * (1.0 + max(clearance, 0.05))
            rot = (normal_w * sz).to_track_quat('Z', 'Y').to_euler()
            key_me = _ball_peg_mesh(
                "MK_Key_M", neck_r, ball_r, neck_len, ball_off)
            key = bpy.data.objects.new("MK_Key_M", key_me)
            key.rotation_euler = rot
            key.location = center_w
            context.collection.objects.link(key)
            fkey_me = _ball_socket_mesh(
                "MK_Key_F", neck_r, ball_r, neck_len, ball_off, socket_depth,
                clearance=clearance)
            fkey = bpy.data.objects.new("MK_Key_F", fkey_me)
            fkey.rotation_euler = rot
            fkey.location = center_w
            context.collection.objects.link(fkey)
            context.view_layer.update()
            hb = ht = db = dt = hh = 0.0
            faces = None
            verts = None
        else:
            H = max(size_ref * h_factor, 1e-5)
            D = max(size_ref * self.depth, 1e-5)
            Wb = max(size_ref * self.width, 1e-5)
            rot = normal_w.to_track_quat('Z', 'Y').to_euler()
            verts = faces = None
            hb = ht = db = dt = hh = H / 2.0

            if shape == 'DOVETAIL':
                Wt = Wb * self.taper
                hb, ht = Wb / 2.0, Wt / 2.0
                db, dt = D / 2.0, (D * self.taper) / 2.0
                hh = H / 2.0
                verts = [
                    (-hb, -db, -hh * sz), (hb, -db, -hh * sz),
                    (hb, db, -hh * sz), (-hb, db, -hh * sz),
                    (-ht, -dt, hh * sz), (ht, -dt, hh * sz),
                    (ht, dt, hh * sz), (-ht, dt, hh * sz),
                ]
                faces = [
                    (0, 1, 2, 3), (4, 5, 6, 7),
                    (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
                ]
                key_me = _mesh_from("MK_Key_M", verts, faces)
                fkey_me = _mesh_from("MK_Key_F", verts, faces)
            elif shape == 'CYLINDER':
                r = max(Wb * 0.5, 1e-5)
                key_me = _cylinder_pin_mesh("MK_Key_M", r, H)
                fkey_me = _cylinder_pin_mesh("MK_Key_F", r, H)
            elif shape == 'BOX':
                key_me = _box_pin_mesh("MK_Key_M", Wb, D, H)
                fkey_me = _box_pin_mesh("MK_Key_F", Wb, D, H)
            elif shape == 'HEX':
                r = max(Wb * 0.5, 1e-5)
                key_me = _hex_pin_mesh("MK_Key_M", r, H)
                fkey_me = _hex_pin_mesh("MK_Key_F", r, H)
            elif shape == 'TAPERED':
                r0 = max(Wb * 0.55, 1e-5)
                r1 = max(r0 * max(self.taper, 0.35), 1e-5)
                key_me = _tapered_pin_mesh("MK_Key_M", r0, r1, H)
                fkey_me = _tapered_pin_mesh("MK_Key_F", r0, r1, H)
            elif shape == 'CROSS':
                key_me = _cross_pin_mesh("MK_Key_M", Wb * 0.35, D, H)
                fkey_me = _cross_pin_mesh("MK_Key_F", Wb * 0.35, D, H)
            elif shape == 'TSLOT':
                key_me = _tslot_pin_mesh(
                    "MK_Key_M", Wb * 0.45, D * 0.55, Wb, D, H)
                fkey_me = _tslot_pin_mesh(
                    "MK_Key_F", Wb * 0.45, D * 0.55, Wb, D, H)
            else:
                # fallback dovetail
                Wt = Wb * self.taper
                hb, ht = Wb / 2.0, Wt / 2.0
                db, dt = D / 2.0, (D * self.taper) / 2.0
                hh = H / 2.0
                verts = [
                    (-hb, -db, -hh * sz), (hb, -db, -hh * sz),
                    (hb, db, -hh * sz), (-hb, db, -hh * sz),
                    (-ht, -dt, hh * sz), (ht, -dt, hh * sz),
                    (ht, dt, hh * sz), (-ht, dt, hh * sz),
                ]
                faces = [
                    (0, 1, 2, 3), (4, 5, 6, 7),
                    (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
                ]
                key_me = _mesh_from("MK_Key_M", verts, faces)
                fkey_me = _mesh_from("MK_Key_F", verts, faces)

            key = bpy.data.objects.new("MK_Key_M", key_me)
            key.rotation_euler = rot
            key.location = center_w
            context.collection.objects.link(key)
            fkey = bpy.data.objects.new("MK_Key_F", fkey_me)
            fkey.rotation_euler = rot
            fkey.location = center_w
            if shape != 'BALL':
                fkey.scale = (fscale, fscale, fscale)
            context.collection.objects.link(fkey)
            context.view_layer.update()

        halves.sort(key=lambda o: (_obj_center_world(o) - center_w).dot(normal_w))
        minus_half, plus_half = halves[0], halves[1]
        flip = getattr(scene, "mk_flip", False)

        # The peg goes on the ACTIVE object, the socket on the other one -
        # the same rule the Recessed Seam follows. This used to be decided by
        # which side of the cut plane a half happened to sit on, which ignores
        # the selection entirely: half the time the peg appeared on the piece
        # the user had not picked, and Flip had to be pressed to put it right.
        act = context.view_layer.objects.active
        if act in halves:
            male_half = act
            female_half = plus_half if act is minus_half else minus_half
        else:
            male_half, female_half = minus_half, plus_half
        if self.swap != flip:
            male_half, female_half = female_half, male_half

        # Point the wedge so its WIDE end is buried in the half that carries
        # it and the NARROW end is the part that sticks out. Tying the wedge's
        # direction to the Flip switch instead of to the chosen half was the
        # bug: whenever Swap and Flip disagreed the peg came out the other way
        # round, wide end waving in the air, and the two halves could not lock.
        toward_male = (_obj_center_world(male_half) - center_w).dot(normal_w)
        sz_key = 1.0 if toward_male < 0 else -1.0
        if self.key_shape == 'DOVETAIL' and verts is not None and faces is not None and sz_key != sz:
            sz = sz_key
            verts = [
                (-hb, -db, -hh * sz), (hb, -db, -hh * sz),
                (hb, db, -hh * sz), (-hb, db, -hh * sz),
                (-ht, -dt, hh * sz), (ht, -dt, hh * sz),
                (ht, dt, hh * sz), (-ht, dt, hh * sz),
            ]
            key.data = _mesh_from("MK_Key_M", verts, faces)
            fkey.data = _mesh_from("MK_Key_F", verts, faces)
            f = 1.0 + clearance
            fkey.scale = (f, f, f)
            context.view_layer.update()

        _apply_boolean(context, male_half, key, 'UNION', self.solver)
        _apply_boolean(context, female_half, fkey, 'DIFFERENCE', self.solver)

        bpy.data.objects.remove(key, do_unlink=True)
        bpy.data.objects.remove(fkey, do_unlink=True)

        smooth_keep_sharp(context, male_half)
        smooth_keep_sharp(context, female_half)

        for o in list(context.selected_objects):
            o.select_set(False)
        male_half.select_set(True)
        female_half.select_set(True)
        context.view_layer.objects.active = male_half
        return {'FINISHED'}


# ----------------------------------------------------------------------------
#  BOOLEAN HELPERS
# ----------------------------------------------------------------------------

def _set_bool_solver(mod, prefer):
    """Assign a boolean solver that exists in this Blender build.
    Names changed across versions: old=FAST/EXACT, new=FLOAT/EXACT/MANIFOLD."""
    if prefer == 'EXACT':
        cands = ['EXACT', 'MANIFOLD', 'FLOAT', 'FAST']
    elif prefer == 'MANIFOLD':
        cands = ['MANIFOLD', 'FLOAT', 'FAST', 'EXACT']
    else:
        cands = ['FLOAT', 'FAST', 'MANIFOLD', 'EXACT']
    for c in cands:
        try:
            mod.solver = c
            return c
        except TypeError:
            continue
    return None


def _obj_center_world(o):
    c = Vector((0.0, 0.0, 0.0))
    for v in o.bound_box:
        c += Vector(v)
    c /= 8.0
    return o.matrix_world @ c


def _apply_boolean(context, target, cutter, op, solver_pref):
    mod = target.modifiers.new("MK_Key", 'BOOLEAN')
    mod.operation = op
    mod.object = cutter
    _set_bool_solver(mod, solver_pref)
    prev = context.view_layer.objects.active
    context.view_layer.objects.active = target
    ok = True
    try:
        # apply on the base mesh, before any other modifiers
        try:
            bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
        except Exception:
            pass
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as ex:
        print("MK boolean apply error:", ex)
        try:
            target.modifiers.remove(mod)
        except Exception:
            pass
        ok = False
    context.view_layer.objects.active = prev
    return ok


# ----------------------------------------------------------------------------
#  STAGE 1 OPERATOR
# ----------------------------------------------------------------------------

def _ray_to_surface(context, obj, mx, my):
    """Screen position -> the point on obj's surface under it, with that
    surface's normal. Returns (world_point, world_normal) or None."""
    try:
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, (mx, my))
        d = view3d_utils.region_2d_to_vector_3d(region, rv3d, (mx, my))
        if o is None or d is None:
            return None
        mwi = obj.matrix_world.inverted()
        ok, loc, nor, _i = obj.ray_cast(mwi @ o,
                                        (mwi.to_3x3() @ d).normalized())
        if not ok:
            return None
        mw = obj.matrix_world
        n_w = (mw.to_3x3() @ nor).normalized()
        return (mw @ loc), n_w
    except Exception:
        return None




def _smooth3d(pts, passes=2):
    """Relax a 3D polyline, keeping its ends pinned."""
    P = [p.copy() for p in pts]
    for _ in range(passes):
        if len(P) < 3:
            break
        Q = [P[0]]
        for i in range(1, len(P) - 1):
            Q.append((P[i - 1] + P[i] * 2.0 + P[i + 1]) / 4.0)
        Q.append(P[-1])
        P = Q
    return P


def build_curve_surface(context, obj, pts, nrms, closed=False):
    """Cutting surface for a line painted on the model.

    Closed loop: a disc spanning the loop - triangles from every edge of the
    line to its centre. That is exactly the cross-section the cut should have,
    and it can never fold through itself.
    Open line: a flat sheet swept along ONE direction (the line's average
    inward normal), carried past both ends so it clears the silhouette.

    Sweeping each point down its OWN normal seems natural but is what broke
    the cut: on a rounded limb those rays converge and cross near the axis, and
    the boolean refuses the self-intersecting sheet that results."""
    try:
        if len(pts) < 3:
            return None
        bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        lo = Vector((min(v[i] for v in bb) for i in range(3)))
        hi = Vector((max(v[i] for v in bb) for i in range(3)))
        diag = (hi - lo).length

        # Size everything off the STROKE, not the whole model. Sweeping the
        # sheet a model-diagonal deep turns it into a curtain that reaches
        # clean through the body and cuts whatever happens to be behind the
        # part you drew on.
        slo = Vector((min(p[i] for p in pts) for i in range(3)))
        shi = Vector((max(p[i] for p in pts) for i in range(3)))
        sdiag = (shi - slo).length
        if sdiag < diag * 1e-3:
            sdiag = diag * 0.05

        bmc = bmesh.new()

        if closed:
            P = list(pts)
            N = list(nrms)
            if (P[-1] - P[0]).length < diag * 1e-4:
                P = P[:-1]
                N = N[:-1]
            if len(P) < 3:
                bmc.free()
                return None

            # Densify: a 15 point loop makes a 15 sided disc, which cuts a
            # visibly faceted outline. Subdividing keeps the cut smooth.
            want = 96
            if len(P) < want:
                D, DN = [], []
                n0 = len(P)
                for i in range(n0):
                    a, b = P[i], P[(i + 1) % n0]
                    na = N[i] if i < len(N) else N[-1]
                    nb = N[(i + 1) % n0] if (i + 1) % n0 < len(N) else N[-1]
                    steps = max(1, int(round(want / float(n0))))
                    for s in range(steps):
                        t = s / float(steps)
                        D.append(a.lerp(b, t))
                        DN.append(na.lerp(nb, t))
                P, N = D, DN

            # Push the rim clear of the surface, because geometry sitting
            # exactly ON the mesh is coincident and the EXACT boolean leaves
            # it uncut. The push has to be RADIAL - straight out within the
            # ring's own plane. Pushing along each point's surface normal is
            # what moved the cut: on a neck or any tapered limb that normal
            # leans along the axis as well as outward, so the whole ring crept
            # up the taper and the cut landed above the line that was drawn.
            cenP = Vector((0.0, 0.0, 0.0))
            for q in P:
                cenP += q
            cenP /= len(P)
            A = np.array([[q.x - cenP.x, q.y - cenP.y, q.z - cenP.z]
                          for q in P])
            try:
                vals, vecs = np.linalg.eigh(A.T @ A)
                v = vecs[:, int(np.argmin(vals))]
                axis = Vector((float(v[0]), float(v[1]), float(v[2])))
                axis = axis.normalized() if axis.length > 1e-9 \
                    else Vector((0.0, 0.0, 1.0))
            except Exception:
                axis = Vector((0.0, 0.0, 1.0))

            out = sdiag * 0.10
            lifted = []
            for q in P:
                rel = q - cenP
                radial = rel - rel.dot(axis) * axis
                if radial.length > 1e-9:
                    lifted.append(q + radial.normalized() * out)
                else:
                    lifted.append(q.copy())
            P = lifted

            ring = [bmc.verts.new(p) for p in P]
            bmc.verts.index_update()
            n = len(ring)
            edges = []
            for i in range(n):
                try:
                    edges.append(bmc.edges.new((ring[i], ring[(i + 1) % n])))
                except ValueError:
                    pass

            # Span the loop with a real triangulation. Fanning every triangle
            # to one centre point looks fine on paper, but a loop taken around
            # a limb is not flat, so that fan folds through itself near the
            # apex and the boolean throws the whole blade out.
            filled = False
            if edges:
                try:
                    res = bmesh.ops.triangle_fill(
                        bmc, edges=edges, use_beauty=True, use_dissolve=False)
                    filled = bool(res.get('geom'))
                except Exception as ex:
                    print("MK triangle fill error:", ex)
            if not filled:
                cen = Vector((0.0, 0.0, 0.0))
                for p in P:
                    cen += p
                cen /= len(P)
                cv = bmc.verts.new(cen)
                bmc.verts.index_update()
                for i in range(n):
                    try:
                        bmc.faces.new((ring[i], ring[(i + 1) % n], cv))
                    except ValueError:
                        pass
        else:
            # ONE sweep direction, so the rays stay parallel and can never
            # fold through each other - the reason per-point normals fail on a
            # rounded limb, where they all converge on the axis. The sheet
            # still contains the painted line exactly, so the cut runs along
            # it. What is measured per point is how FAR to go: each ray stops
            # just past where the part ends, so the sheet cannot travel on and
            # meet the body somewhere else and cut there instead.
            d = Vector((0.0, 0.0, 0.0))
            for nn in nrms:
                d += nn
            if d.length < 1e-9:
                d = Vector((0.0, 0.0, 1.0))
            d.normalize()

            P = list(pts)
            ext = sdiag * 0.35
            t0 = P[0] - P[1]
            t1 = P[-1] - P[-2]
            if t0.length > 1e-9:
                P.insert(0, P[0] + t0.normalized() * ext)
            if t1.length > 1e-9:
                P.append(P[-1] + t1.normalized() * ext)

            mwi = obj.matrix_world.inverted()
            d_l = (mwi.to_3x3() @ (-d)).normalized()
            deep = []
            for p in P:
                start = mwi @ (p - d * (sdiag * 0.001))
                ok_h, loc, _n, _i = obj.ray_cast(start, d_l)
                t = (loc - start).length if ok_h else 0.0
                if t <= 1e-9:
                    t = sdiag * 0.5
                deep.append(min(t * 1.2 + sdiag * 0.04, diag * 0.9))
            # one shared depth, taken from the thickest sample, keeps the far
            # edge tidy while still stopping short of anything else
            depth = max(deep)

            near = [bmc.verts.new(p + d * (sdiag * 0.04)) for p in P]
            far = [bmc.verts.new(p - d * depth) for p in P]
            bmc.verts.index_update()
            for i2 in range(len(P) - 1):
                try:
                    bmc.faces.new((near[i2], near[i2 + 1],
                                   far[i2 + 1], far[i2]))
                except ValueError:
                    pass

        if not bmc.faces:
            bmc.free()
            return None
        bmesh.ops.recalc_face_normals(bmc, faces=bmc.faces[:])
        me = bpy.data.meshes.new("MK_Surface")
        bmc.to_mesh(me)
        bmc.free()
        me.validate(verbose=False)
        me.update()
        if not len(me.polygons):
            bpy.data.meshes.remove(me)
            return None
        surf = bpy.data.objects.new("MK_Surface", me)
        surf.show_wire = True
        context.collection.objects.link(surf)
        context.view_layer.update()
        return surf
    except Exception as ex:
        print("MK curve surface error:", ex)
        return None


class MK_OT_line_slice(bpy.types.Operator):
    bl_idname = "mk.line_slice"
    bl_label = "Нарисовать линию реза"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Выберите MESH-объект")
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Используйте в 3D Viewport")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self.obj = obj
        self.pen = getattr(context.scene, "mk_pen", False)
        self.freehand = (getattr(context.scene, "mk_freehand", False)
                         and not self.pen)
        # Multi Cut is not an alternative to the three drawing styles, it
        # changes WHEN the cut happens: strokes are stored and applied
        # together, so a piece held in several places can come free.
        self.multi = getattr(context.scene, "mk_multi", False)
        self.points = []
        self.dragging = False
        self._handle = None
        # A freehand stroke is not cut straight away: it becomes an editable
        # curve first, so the shape can be adjusted before anything is
        # committed. Straight-line cuts are unaffected and still cut on
        # release.
        self.phase = 'DRAW'
        self.raw = []           # every sample the pen laid on the surface
        self.rawn = []
        self.ctrl = []          # the handles shown: raw resampled to N points
        self.cnrm = []
        self.drag_i = -1
        self.closed = False
        self.manual = False     # a handle was nudged, so stop resampling
        self.stroke_starts = []
        self.edit = False       # TAB switches between painting and editing
        self.hover = -1
        self.n_drag = False     # dragging a point along its own normal
        self._rev = 0           # bumped whenever the handles change
        self.ins_pt = None      # where CTRL would drop a new point
        self._curve = []
        self._curve_key = None
        self._vis = None
        self._vis_k = None
        self._pending = 0

        if bpy.app.version >= (4, 0, 0):
            self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            self._3d = True
        else:
            self.shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
            self._3d = False

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_px, (), 'WINDOW', 'POST_PIXEL')
        # a slow tick, purely so moving the Points slider re-spaces the
        # handles straight away instead of waiting for the next mouse move
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window.cursor_modal_set('KNIFE')
        if self.pen:
            context.workspace.status_text_set(self._status())
        elif self.multi:
            context.workspace.status_text_set(self._multi_status())
        elif self.freehand:
            context.workspace.status_text_set(
                "LMB drag: draw freehand cut   |   ESC / RMB: cancel")
        else:
            context.workspace.status_text_set(
                "LMB drag: draw straight line cut   |   ESC / RMB: cancel")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()

        if self.pen:
            return self._modal_paint(context, event)

        # Multi Cut keeps the tool running between strokes, so navigation and
        # the side panel have to stay reachable - otherwise the second stroke
        # can never be aimed at the far side of the model.
        if self.multi and not self.dragging:
            r = context.region
            if r is not None:
                mx = event.mouse_region_x
                my = event.mouse_region_y
                if mx < 0 or my < 0 or mx > r.width or my > r.height:
                    return {'PASS_THROUGH'}
            if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                              'TRACKPADPAN', 'TRACKPADZOOM'} \
                    or event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT'}:
                return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            if self.dragging:
                p = (event.mouse_region_x, event.mouse_region_y)
                if self.freehand:
                    # original freehand: collect the screen stroke and cut on
                    # release, exactly as it always did
                    if not self.points or \
                            (Vector(p) - Vector(self.points[-1])).length >= 3.0:
                        self.points.append(p)
                elif self.points:
                    self.points = [self.points[0], p]

        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.dragging = True
                self.points = [(event.mouse_region_x, event.mouse_region_y)]
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                if not self.dragging:
                    # stray release from the UI button click - ignore it
                    return {'RUNNING_MODAL'}
                self.dragging = False
                p = (event.mouse_region_x, event.mouse_region_y)
                if self.freehand:
                    self.points.append(p)
                elif self.points:
                    self.points = [self.points[0], p]
                pts = list(self.points)
                if self.multi:
                    self._stash(context, pts)
                    self.points = []
                    return {'RUNNING_MODAL'}
                self.finish(context)
                self._run_cut(context, pts)
                return {'FINISHED'}

        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if self.multi:
                self.finish(context)
                if _pending_count():
                    bpy.ops.mk.apply_multi('INVOKE_DEFAULT')
                else:
                    self.report({'WARNING'}, "Сначала нарисуйте хотя бы один рез")
                return {'FINISHED'}

        elif event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
            if self.multi:
                _pending_pop(context)
                context.workspace.status_text_set(self._multi_status())
                return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    # ---- freehand: paint the line onto the model, like a pen -------------

    def _modal_paint(self, context, event):
        # Hand every event outside the viewport back to Blender. A modal
        # operator otherwise swallows clicks on the side panel too, so the
        # Points slider could not be touched while the pen was running - which
        # reads exactly like Blender having locked up.
        if not self.dragging and self.drag_i < 0:
            r = context.region
            if r is not None:
                mx = event.mouse_region_x
                my = event.mouse_region_y
                if mx < 0 or my < 0 or mx > r.width or my > r.height:
                    return {'PASS_THROUGH'}

        # respond to the Points slider wherever we are
        # the Points slider takes effect straight away, whether the pen is
        # mid-stroke or the line is already finished
        want = getattr(context.scene, "mk_pen_points", 16)
        if want != getattr(self, "_last_want", None):
            self._last_want = want
            self._pending = 0
            self._rebuild_ctrl(context)
            if context.area:
                context.area.tag_redraw()
        if event.type == 'TIMER':
            return {'RUNNING_MODAL'}
        if self.edit:
            return self._modal_edit(context, event)
        if event.type == 'TAB' and event.value == 'PRESS':
            self._rebuild_ctrl(context)
            if len(self.ctrl) >= 2:
                self.edit = True
                self.dragging = False
                context.window.cursor_modal_set('SCROLL_XY')
                context.workspace.status_text_set(self._status())
            return {'RUNNING_MODAL'}

        """Режим от руки: зажмите ЛКМ — линия ложится на
        surface under the cursor. Let go, orbit to the far side, and carry on -
        that is how a cut is taken right around an arm. Nothing is cut until
        ENTER."""
        # navigation stays live between strokes, so the model can be turned
        if not self.dragging and (
                event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE',
                               'WHEELDOWNMOUSE', 'TRACKPADPAN',
                               'TRACKPADZOOM'}
                or (event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT'})):
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            if self.dragging and not self.closed:
                p = Vector((event.mouse_region_x, event.mouse_region_y))
                if not self.points or \
                        (p - Vector(self.points[-1])).length >= 6.0:
                    hit = _ray_to_surface(context, self.obj, p.x, p.y)
                    if hit is not None:
                        # Coming back to where the line started means the cut
                        # goes right around the part. Inside the snap distance
                        # the end is pulled onto the first point exactly and
                        # the two are welded, so the loop really is closed -
                        # not merely close, which looked like a gap and left
                        # the user guessing.
                        if len(self.raw) > 8:
                            s0 = self._screen(context, self.raw[0])
                            if s0 is not None and \
                                    (Vector((s0.x, s0.y)) - p).length \
                                    <= SNAP_PX:
                                self.points.append((s0.x, s0.y))
                                self._add_sample(
                                    context, (self.raw[0].copy(),
                                              self.rawn[0].copy()))
                                # the visible handle lands on the first one too
                                if len(self.ctrl) >= 2:
                                    self.ctrl[-1] = self.ctrl[0].copy()
                                    self.cnrm[-1] = self.cnrm[0].copy()
                                    self._rev += 1
                                self.closed = True
                                self.dragging = False
                                self.report({'INFO'},
                                            "Контур замкнут и сварен")
                                return {'RUNNING_MODAL'}
                        self.points.append((p.x, p.y))
                        self._add_sample(context, hit)
            elif self.drag_i >= 0:
                hit = _ray_to_surface(context, self.obj,
                                      event.mouse_region_x,
                                      event.mouse_region_y)
                if hit is not None:
                    self.ctrl[self.drag_i] = hit[0]
                    self.cnrm[self.drag_i] = hit[1]
                    self._rev += 1
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                m = Vector((event.mouse_region_x, event.mouse_region_y))
                if event.ctrl:
                    best, bd = -1, 20.0
                    for i, p in enumerate(self.ctrl):
                        s = self._screen(context, p)
                        if s is None:
                            continue
                        d = (Vector((s.x, s.y)) - m).length
                        if d < bd:
                            bd, best = d, i
                    if best >= 0:
                        # from here on the handles are the user's, so they are
                        # no longer thrown away and rebuilt from the raw pen
                        self.manual = True
                    self.drag_i = best
                    return {'RUNNING_MODAL'}
                if self.closed:
                    return {'RUNNING_MODAL'}
                self.dragging = True
                self.stroke_starts.append(len(self.raw))
                # Carry on from where the line stopped. Starting a fresh
                # stroke somewhere else would leave a straight jump across the
                # model, so the gap between the last point and the new click is
                # walked over the surface first and the line stays one piece.
                if self.ctrl:
                    last = self._screen(context, self.ctrl[-1])
                    if last is not None:
                        A = Vector((last.x, last.y))
                        gap = (m - A).length
                        if gap > 8.0:
                            steps = min(int(gap // 8.0), 200)
                            for s in range(1, steps + 1):
                                q = A + (m - A) * (s / float(steps))
                                h = _ray_to_surface(context, self.obj,
                                                    q.x, q.y)
                                if h is not None:
                                    self.points.append((q.x, q.y))
                                    self._add_sample(context, h)
                hit = _ray_to_surface(context, self.obj, m.x, m.y)
                if hit is not None:
                    self.points.append((m.x, m.y))
                    self._add_sample(context, hit)
            else:
                if self.dragging and self._pending:
                    self._pending = 0
                    self._rebuild_ctrl(context)
                self.dragging = False
                self.drag_i = -1
            return {'RUNNING_MODAL'}

        if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
            if not self.stroke_starts and self.multi:
                # nothing painted yet, so take back the last STORED cut
                _pending_pop(context)
                context.workspace.status_text_set(self._status())
                return {'RUNNING_MODAL'}
            if self.stroke_starts:
                k = self.stroke_starts.pop()
                del self.raw[k:]
                del self.rawn[k:]
                self.points = self.points[:k]
                self.closed = False
                # start over from the raw stroke, so the slider works again
                self.manual = False
                self._step = None
                self._last_want = None
                self._rebuild_ctrl(context)
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self._pending = 0
            self._rebuild_ctrl(context)
            if self.multi:
                # The pen deliberately bridges each new stroke onto the end of
                # the last one, so one painted line is one cut however many
                # strokes went into it. ENTER therefore banks the line and
                # clears the pen for the next one; ENTER on an empty pen (or
                # SHIFT+ENTER) applies everything banked so far.
                stored = False
                if len(self.ctrl) >= 3:
                    pts, nrm, closed = self._finalise_curve()
                    surf = build_curve_surface(context, self.obj, pts, nrm,
                                               closed)
                    if surf is None:
                        self.report({'WARNING'},
                                    "Не удалось построить поверхность реза из этой линии")
                        return {'RUNNING_MODAL'}
                    _pending_add_surf(context, self.obj, surf, closed, pts)
                    self._reset_line(context)
                    stored = True
                    self.report({'INFO'}, "Рез %d сохранён — рисуйте следующий "
                                          "ENTER снова — применить"
                                % _pending_count())
                if stored and not event.shift:
                    context.workspace.status_text_set(self._status())
                    return {'RUNNING_MODAL'}
                if not _pending_count():
                    self.report({'WARNING'},
                                "Сначала проведите линию по модели")
                    return {'RUNNING_MODAL'}
                self.finish(context)
                bpy.ops.mk.apply_multi('INVOKE_DEFAULT')
                return {'FINISHED'}
            if len(self.ctrl) < 3:
                self.report({'WARNING'},
                            "Сначала проведите линию по модели")
                return {'RUNNING_MODAL'}
            pts, nrm, closed = self._finalise_curve()
            obj = self.obj
            self.finish(context)
            self._run_curve_cut(context, obj, pts, nrm, closed)
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _stash(self, context, pts):
        """Store one straight-line or freehand stroke instead of cutting."""
        if len(pts) < 2:
            return
        if self.freehand:
            surf = build_stroke_surface(context, self.obj, pts)
            if surf is None:
                self.report({'WARNING'},
                            "Не удалось построить поверхность реза из этого штриха")
                return
            _pending_add_surf(context, self.obj, surf, False, None,
                              _ghost_pierce(context, self.obj, pts))
        else:
            a, b = Vector(pts[0]), Vector(pts[-1])
            if (b - a).length <= 2:
                return
            if not _pending_add_line(context, self.obj, a, b):
                self.report({'WARNING'}, "Линия слишком короткая для сохранения")
                return
        self.report({'INFO'}, "Рез %d сохранён" % _pending_count())
        context.workspace.status_text_set(self._multi_status())
        _redraw_areas(context)

    def _multi_status(self):
        return ("MULTI CUT  |  %d stored   draw another   MMB: orbit   "
                "BACKSPACE: undo last   ENTER: apply   ESC: leave"
                % _pending_count())

    def _reset_line(self, context):
        """Clear the pen so the next separate cut can be painted."""
        self.points = []
        self.raw = []
        self.rawn = []
        self.ctrl = []
        self.cnrm = []
        self.stroke_starts = []
        self.closed = False
        self.manual = False
        self.dragging = False
        self.drag_i = -1
        self.hover = -1
        self.n_drag = False
        self.edit = False
        self.ins_pt = None
        self._step = None
        self._last_want = None
        self._curve = []
        self._curve_key = None
        self._vis = None
        self._vis_k = None
        self._pending = 0
        self._rev += 1
        try:
            context.window.cursor_modal_set('KNIFE')
        except Exception:
            pass

    def _finalise_curve(self):
        """Smooth the handles into the curve the cut is built from. A loop
        closed by the snap is welded end to end here."""
        pts = list(self.ctrl)
        nrm = list(self.cnrm)
        closed = self.closed
        if not closed and len(pts) > 6:
            span = sum((pts[i] - pts[i - 1]).length
                       for i in range(1, len(pts)))
            closed = span > 1e-9 and \
                (pts[-1] - pts[0]).length < span * 0.12
        if closed and len(pts) > 2:
            # weld: make the last point the first one exactly
            pts[-1] = pts[0].copy()
            nrm[-1] = nrm[0].copy()
        # Densify by walking straight between the handles - no relaxation.
        # Averaging neighbours pulls a curved line toward its chord, so the
        # dip in a stroke drawn across the chest rose by centimetres and the
        # cut came out above where it was painted.
        dense, densen = [], []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            na = nrm[min(i, len(nrm) - 1)]
            nb = nrm[min(i + 1, len(nrm) - 1)]
            steps = 12
            for s in range(steps):
                t = s / float(steps)
                dense.append(a.lerp(b, t))
                densen.append(na.lerp(nb, t))
        dense.append(pts[-1])
        densen.append(nrm[-1])

        # No snapping to the surface here. Every handle already came off the
        # model through a ray cast, so it is on the surface to begin with. The
        # in-between points are chords, which dip a fraction inside a curved
        # part - and pulling those back to the NEAREST surface point slides
        # them up the slope of a neck or a limb. Repeated round a ring, that
        # is what walked the whole cut upward from where it was drawn.
        return dense, densen, closed

    # ---- freehand: edit the curve before cutting -----------------------

    def _screen(self, context, p):
        return view3d_utils.location_3d_to_region_2d(
            context.region, context.region_data, p)

    def _visible(self, context, pts):
        """Which of these points the model is not hiding.

        The line is drawn in 2D over the viewport, so it has no depth of its
        own and would otherwise show straight through the model - draw a ring
        round a leg, turn the view, and the far side of the ring is still
        sitting on top of the near side. A short ray from the eye to each
        point settles it: if the surface is reached first, that point is
        behind and is left out."""
        region = context.region
        rv3d = context.region_data
        obj = self.obj
        try:
            mw = obj.matrix_world
            mwi = mw.inverted()
            n3i = mwi.to_3x3()
            bb = [mw @ Vector(c) for c in obj.bound_box]
            lo = Vector((min(v[i] for v in bb) for i in range(3)))
            hi = Vector((max(v[i] for v in bb) for i in range(3)))
            eps = (hi - lo).length * 0.002
            out = []
            for p in pts:
                s = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
                if s is None:
                    out.append(False)
                    continue
                o = view3d_utils.region_2d_to_origin_3d(region, rv3d, (s.x,
                                                                       s.y))
                if o is None:
                    out.append(True)
                    continue
                d = p - o
                dist = d.length
                if dist < 1e-9:
                    out.append(True)
                    continue
                ok, loc, _n, _i = obj.ray_cast(
                    mwi @ o, (n3i @ d).normalized())
                if not ok:
                    out.append(True)
                    continue
                out.append(((mw @ loc) - o).length >= dist - eps)
            return out
        except Exception as ex:
            print("MK visibility error:", ex)
            return [True] * len(pts)

    def _vis_key(self, context):
        rv = context.region_data
        try:
            m = rv.view_matrix
            return (self._rev, len(self.ctrl),
                    tuple(round(v, 4) for r in m for v in r),
                    round(rv.view_distance, 4))
        except Exception:
            return (self._rev, len(self.ctrl))

    def _status(self):
        tail = ""
        if getattr(self, "multi", False):
            tail = "   [MULTI: %d stored, ENTER on an empty line applies]" \
                % _pending_count()
        if self.edit:
            return ("EDIT  |  drag: move on surface   SHIFT+drag: along "
                    "normal   CTRL+click line: add point   X: delete   "
                    "TAB: paint   ENTER: cut   ESC: cancel" + tail)
        return ("PAINT  |  hold LMB to draw on the model   MMB: orbit   "
                "BACKSPACE: undo stroke   TAB: edit points   ENTER: cut   "
                "ESC: cancel" + tail)

    def _pick(self, context, m, tol=20.0):
        best, bd = -1, tol
        for i, p in enumerate(self.ctrl):
            s = self._screen(context, p)
            if s is None:
                continue
            d = (Vector((s.x, s.y)) - m).length
            if d < bd:
                bd, best = d, i
        return best

    def _insert_spot(self, context, m, tol=25.0):
        """Nearest point ON the line to the cursor: (screen_pos, index, t)."""
        if len(self.ctrl) < 2:
            return None
        best = None
        for i in range(len(self.ctrl) - 1):
            a = self._screen(context, self.ctrl[i])
            b = self._screen(context, self.ctrl[i + 1])
            if a is None or b is None:
                continue
            A = Vector((a.x, a.y))
            B = Vector((b.x, b.y))
            ab = B - A
            L2 = ab.length_squared
            t = 0.5 if L2 < 1e-9 else max(0.0, min(1.0, (m - A).dot(ab) / L2))
            q = A + ab * t
            d = (q - m).length
            if best is None or d < best[0]:
                best = (d, q, i, t)
        if best is None or best[0] > tol:
            return None
        return (best[1], best[2], best[3])

    def _insert_point(self, context, m):
        """Add a point where the line was clicked, between the two handles
        that segment belongs to, so the curve gains detail exactly there."""
        spot = self._insert_spot(context, m)
        if spot is None:
            return False
        _q, best, bt = spot
        hit = _ray_to_surface(context, self.obj, m.x, m.y)
        if hit is not None:
            p, n = hit
        else:
            p = self.ctrl[best].lerp(self.ctrl[best + 1], bt)
            n = self.cnrm[best].lerp(self.cnrm[best + 1], bt)
        self.manual = True
        self.ctrl.insert(best + 1, p)
        self.cnrm.insert(best + 1, n)
        self._rev += 1
        return True

    def _modal_edit(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                          'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            m = Vector((event.mouse_region_x, event.mouse_region_y))
            # with CTRL held, show exactly where a new point would go, so the
            # feature is visible instead of having to be known about
            self.ins_pt = None
            if event.ctrl and self.drag_i < 0:
                spot = self._insert_spot(context, m)
                if spot is not None:
                    self.ins_pt = spot[0]
            if self.drag_i >= 0:
                if self.n_drag:
                    # slide the point along its own surface normal, so it can
                    # sit proud of the model or sink into it
                    s = self._screen(context, self.ctrl[self.drag_i])
                    if s is not None:
                        bb = [self.obj.matrix_world @ Vector(c)
                              for c in self.obj.bound_box]
                        lo = Vector((min(v[i] for v in bb) for i in range(3)))
                        hi = Vector((max(v[i] for v in bb) for i in range(3)))
                        scale = (hi - lo).length * 0.0015
                        dy = m.y - getattr(self, "_ny", m.y)
                        self.ctrl[self.drag_i] = (
                            self.ctrl[self.drag_i]
                            + self.cnrm[self.drag_i] * (dy * scale))
                        self._rev += 1
                    self._ny = m.y
                else:
                    hit = _ray_to_surface(context, self.obj, m.x, m.y)
                    if hit is not None:
                        self.ctrl[self.drag_i] = hit[0]
                        self.cnrm[self.drag_i] = hit[1]
                        # a welded loop stays welded: the two ends are one
                        # point, so moving either has to move both
                        if self.closed and len(self.ctrl) > 2 and \
                                self.drag_i in (0, len(self.ctrl) - 1):
                            other = -1 if self.drag_i == 0 else 0
                            self.ctrl[other] = hit[0].copy()
                            self.cnrm[other] = hit[1].copy()
                        self._rev += 1
            else:
                self.hover = self._pick(context, m)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                m = Vector((event.mouse_region_x, event.mouse_region_y))
                if event.ctrl:
                    if self._insert_point(context, m):
                        self.ins_pt = None
                        self.report({'INFO'}, "Точка добавлена (%d всего)"
                                    % len(self.ctrl))
                        return {'RUNNING_MODAL'}
                    self.report({'WARNING'}, "Кликните по линии, чтобы добавить точку")
                    return {'RUNNING_MODAL'}
                i = self._pick(context, m)
                if i >= 0:
                    self.manual = True
                    self.drag_i = i
                    self.n_drag = bool(event.shift)
                    self._ny = m.y
            else:
                # No re-spacing here. Baking on every drag re-ran the
                # resampler, which quietly slid the handles back to even
                # spacing - it looked like new points appearing and made
                # deleting a point do nothing at all. The line is only folded
                # back into the stroke when edit mode is left.
                self.drag_i = -1
                self.n_drag = False
            return {'RUNNING_MODAL'}

        if event.type in {'X', 'DEL', 'BACK_SPACE'} and event.value == 'PRESS':
            # take the point under the cursor; fall back to a fresh pick so a
            # stale hover never blocks the delete
            i = self.hover
            if not (0 <= i < len(self.ctrl)):
                i = self._pick(context, Vector((event.mouse_region_x,
                                                event.mouse_region_y)), 25.0)
            if 0 <= i < len(self.ctrl):
                if len(self.ctrl) <= 3:
                    self.report({'WARNING'},
                                "Нужно минимум три точки")
                    return {'RUNNING_MODAL'}
                self.manual = True
                del self.ctrl[i]
                del self.cnrm[i]
                if self.closed and len(self.ctrl) > 2:
                    self.ctrl[-1] = self.ctrl[0].copy()
                    self.cnrm[-1] = self.cnrm[0].copy()
                self._rev += 1
                self.hover = -1
                self.report({'INFO'}, "Точка удалена (%d осталось)"
                            % len(self.ctrl))
            return {'RUNNING_MODAL'}

        if event.type == 'TAB' and event.value == 'PRESS':
            self.edit = False
            self.drag_i = -1
            # Leave the line exactly as edited. Re-spacing it here shifted the
            # shape and put back points that had been deleted, which is not
            # what "go back to drawing" should do.
            self._step = self._ctrl_spacing()
            context.window.cursor_modal_set('KNIFE')
            context.workspace.status_text_set(self._status())
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            pts, nrm, closed = self._finalise_curve()
            obj = self.obj
            self.finish(context)
            self._run_curve_cut(context, obj, pts, nrm, closed)
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _ctrl_spacing(self):
        """The gap the current points sit at, so anything drawn afterwards
        carries on at the same density."""
        if len(self.ctrl) < 2:
            return None
        span = 0.0
        for i in range(1, len(self.ctrl)):
            span += (self.ctrl[i] - self.ctrl[i - 1]).length
        return span / (len(self.ctrl) - 1) if span > 1e-12 else None

    def _add_sample(self, context, hit, force=False):
        """Record a point the pen just laid down.

        Re-spacing the whole line costs a pass over every sample, so doing it
        for each one turns drawing into an O(n^2) crawl. It is done every few
        samples instead, and always at the end of a stroke."""
        self.raw.append(hit[0])
        self.rawn.append(hit[1])

        if self.manual:
            # The handles are the user's now, so they are never rebuilt. The
            # line still grows, one point at a time, at the spacing the
            # existing points already use.
            step = getattr(self, "_step", None) or self._ctrl_spacing()
            if step is None or not self.ctrl or \
                    (hit[0] - self.ctrl[-1]).length >= step:
                self.ctrl.append(hit[0])
                self.cnrm.append(hit[1])
                self._rev += 1
            return

        self._pending += 1
        if force or self._pending >= 4:
            self._pending = 0
            self._rebuild_ctrl(context)

    def _rebuild_ctrl(self, context):
        """Thin the painted samples down to the number of handles asked for.
        The pen lays down far more points than anyone wants to grab, so the
        handles are spread evenly along the line's length - raising the count
        gives finer control, lowering it gives a cleaner curve."""
        if self.manual:
            return
        want = getattr(context.scene, "mk_pen_points", 16)
        P, N = self.raw, self.rawn
        if len(P) < 2:
            self.ctrl = list(P)
            self.cnrm = list(N)
            self._rev += 1
            return
        d = [0.0]
        for i in range(1, len(P)):
            d.append(d[-1] + (P[i] - P[i - 1]).length)
        total = d[-1]
        if total <= 1e-12:
            self.ctrl = [P[0]]
            self.cnrm = [N[0]]
            self._rev += 1
            return
        want = max(3, min(int(want), len(P)))
        out, outn = [], []
        j = 0
        for k in range(want):
            t = total * k / (want - 1)
            while j < len(d) - 2 and d[j + 1] < t:
                j += 1
            seg = (d[j + 1] - d[j]) or 1.0
            f = (t - d[j]) / seg
            out.append(P[j].lerp(P[j + 1], f))
            outn.append(N[j].lerp(N[j + 1], f)
                        if j + 1 < len(N) else N[min(j, len(N) - 1)])
        self.ctrl = out
        self.cnrm = outn
        self._rev += 1

    def curve_points(self):
        """The handles relaxed into a flowing curve, cached.

        This runs on every redraw, so it must stay cheap: the points already
        came off the surface via ray casts, and snapping each one back to the
        mesh here was enough to stall the viewport while drawing. That snap is
        done once, at cut time, where it actually matters."""
        # Key on the revision counter alone. Keying on the point count as
        # well hid every move: dragging a handle does not change how many
        # there are, so the stale curve was handed back and nothing appeared
        # to happen until a point was deleted.
        key = self._rev
        if self._curve_key == key:
            return self._curve
        # straight between the handles, matching what the cut will follow
        out = list(self.ctrl)
        self._curve_key = key
        self._curve = out
        return out

    def _run_curve_cut(self, context, obj, pts, nrm, closed):
        surf = build_curve_surface(context, obj, pts, nrm, closed)
        if surf is None:
            self.report({'WARNING'},
                        "Не удалось построить поверхность реза из этой линии")
            return
        faces = len(surf.data.polygons)
        ok = do_surface_cut(context, obj, surf)
        try:
            if surf.name in bpy.data.objects:
                bpy.data.objects.remove(surf, do_unlink=True)
        except Exception:
            pass

        if not ok:
            # A blade that only grazes the mesh leaves it in one piece. Give
            # the surface some room either side and try once more before
            # giving up - a hair-thin sheet on a dense model is easy to miss.
            surf2 = build_curve_surface(context, obj, pts, nrm, closed)
            if surf2 is not None:
                try:
                    bb = [obj.matrix_world @ Vector(c)
                          for c in obj.bound_box]
                    lo = Vector((min(v[i] for v in bb) for i in range(3)))
                    hi = Vector((max(v[i] for v in bb) for i in range(3)))
                    grow = (hi - lo).length * 0.02
                    bm2 = bmesh.new()
                    bm2.from_mesh(surf2.data)
                    bmesh.ops.solidify(bm2, geom=bm2.faces[:], thickness=grow)
                    bm2.to_mesh(surf2.data)
                    bm2.free()
                    surf2.data.update()
                except Exception as ex:
                    print("MK blade thicken error:", ex)
                ok = do_surface_cut(context, obj, surf2)
                try:
                    if surf2.name in bpy.data.objects:
                        bpy.data.objects.remove(surf2, do_unlink=True)
                except Exception:
                    pass
        if ok:
            self.report({'INFO'}, "Рез готов (%s линия, %d граней)"
                        % ("closed" if closed else "open", faces))
        else:
            self.report({'WARNING'},
                        "Сбой реза — %s линия, %d граней. Линия должна "
                        "пересечь деталь насквозь."
                        % ("closed" if closed else "open", faces))

    def _run_cut(self, context, pts):
        if len(pts) < 2:
            return
        if self.freehand:
            # one stroke does it all: build surface -> cut by it -> remove it
            surf = build_stroke_surface(context, self.obj, pts)
            ok = False
            if surf is not None:
                ok = do_surface_cut(context, self.obj, surf)
                try:
                    if surf.name in bpy.data.objects:
                        bpy.data.objects.remove(surf, do_unlink=True)
                except Exception:
                    pass
            if not ok:
                self.report({'WARNING'},
                            "Сбой реза от руки — линия должна пересечь деталь насквозь")
        else:
            s, e = Vector(pts[0]), Vector(pts[-1])
            if (e - s).length > 2:
                if not do_slice(context, self.obj, s, e):
                    self.report({'WARNING'},
                                "Линия должна пересечь деталь насквозь. Попробуйте снова")

    def finish(self, context):
        t = getattr(self, "_timer", None)
        if t is not None:
            try:
                context.window_manager.event_timer_remove(t)
            except Exception:
                pass
            self._timer = None
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()

    def _px(self, p):
        return (p[0], p[1], 0.0) if self._3d else (p[0], p[1])

    def _seg(self, pts):
        out = []
        for i in range(len(pts) - 1):
            out.append(self._px(pts[i]))
            out.append(self._px(pts[i + 1]))
        return out

    def draw_px(self):
      try:
        gpu.state.blend_set('ALPHA')
        self.shader.bind()

        if self.pen:
            ctx = bpy.context
            curve = self.curve_points()

            # hide whatever the model is standing in front of, so the far side
            # of a ring does not draw over the near side
            key = self._vis_key(ctx)
            if self._vis_k != key or self._vis is None \
                    or len(self._vis) != len(curve) + len(self.ctrl):
                self._vis_k = key
                self._vis = self._visible(ctx, list(curve) + list(self.ctrl))
            vis_c = self._vis[:len(curve)]
            vis_h = self._vis[len(curve):]

            scr = [self._screen(ctx, p) for p in curve]
            segs = []
            for i in range(len(curve) - 1):
                if not (vis_c[i] and vis_c[i + 1]):
                    continue
                a, b = scr[i], scr[i + 1]
                if a is None or b is None:
                    continue
                segs.append(self._px((a.x, a.y)))
                segs.append(self._px((b.x, b.y)))
            if segs:
                gpu.state.line_width_set(3.0)
                self.shader.uniform_float("color", (1.0, 0.15, 0.15, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": segs}).draw(self.shader)
            # handles, as small squares so they read at any zoom
            r = 4.0
            quads = []
            hs = []
            for hi_, p in enumerate(self.ctrl):
                s = self._screen(ctx, p)
                if hi_ < len(vis_h) and not vis_h[hi_]:
                    s = None          # behind the model
                hs.append(s)
                if s is None:
                    continue
                a = (s.x - r, s.y - r)
                b = (s.x + r, s.y - r)
                c = (s.x + r, s.y + r)
                d = (s.x - r, s.y + r)
                for u, v in ((a, b), (b, c), (c, d), (d, a)):
                    quads.append(self._px(u))
                    quads.append(self._px(v))
            if quads:
                gpu.state.line_width_set(2.0)
                self.shader.uniform_float("color", (1.0, 0.9, 0.2, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": quads}).draw(self.shader)
            # the first point, so it is obvious where to come back to. It
            # swells and turns green as the pen enters the snap distance, so
            # the pull is visible before the button is released.
            near_snap = False
            try:
                if not self.closed and len(self.ctrl) > 3 and self.points:
                    s0 = hs[0]
                    if s0 is not None:
                        m = Vector(self.points[-1])
                        near_snap = (Vector((s0.x, s0.y)) - m).length <= SNAP_PX
            except Exception:
                near_snap = False
            if len(self.ctrl) > 3 and hs and hs[0] is not None:
                s = hs[0]
                r3 = 14.0 if (near_snap or self.closed) else 9.0
                a = (s.x - r3, s.y - r3)
                b = (s.x + r3, s.y - r3)
                c = (s.x + r3, s.y + r3)
                dd = (s.x - r3, s.y + r3)
                ring = []
                for u, v in ((a, b), (b, c), (c, dd), (dd, a)):
                    ring.append(self._px(u))
                    ring.append(self._px(v))
                self.shader.uniform_float(
                    "color", (0.2, 1.0, 0.4, 1.0)
                    if (self.closed or near_snap) else (0.3, 0.7, 1.0, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": ring}).draw(self.shader)

            # the point under the cursor, so it is clear what X will remove
            if self.edit and 0 <= self.hover < len(hs) \
                    and self.hover != self.drag_i \
                    and hs[self.hover] is not None:
                s = hs[self.hover]
                r4 = 7.0
                a = (s.x - r4, s.y - r4)
                b = (s.x + r4, s.y - r4)
                c = (s.x + r4, s.y + r4)
                dd = (s.x - r4, s.y + r4)
                hv = []
                for u, v in ((a, b), (b, c), (c, dd), (dd, a)):
                    hv.append(self._px(u))
                    hv.append(self._px(v))
                self.shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": hv}).draw(self.shader)

            # a plus on the line marks where CTRL+click would add a point
            if self.edit and self.ins_pt is not None:
                q = self.ins_pt
                r5 = 7.0
                cross = []
                for u, v in (((q.x - r5, q.y), (q.x + r5, q.y)),
                             ((q.x, q.y - r5), (q.x, q.y + r5))):
                    cross.append(self._px(u))
                    cross.append(self._px(v))
                gpu.state.line_width_set(2.0)
                self.shader.uniform_float("color", (0.3, 1.0, 0.5, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": cross}).draw(self.shader)

            if 0 <= self.drag_i < len(hs) and hs[self.drag_i] is not None:
                s = hs[self.drag_i]
                r2 = 7.0
                a = (s.x - r2, s.y - r2)
                b = (s.x + r2, s.y - r2)
                c = (s.x + r2, s.y + r2)
                d = (s.x - r2, s.y + r2)
                hl = []
                for u, v in ((a, b), (b, c), (c, d), (d, a)):
                    hl.append(self._px(u))
                    hl.append(self._px(v))
                self.shader.uniform_float("color", (0.2, 1.0, 0.4, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": hl}).draw(self.shader)
        else:
            pts = self.points
            if len(pts) >= 2:
                gpu.state.line_width_set(4.0)
                self.shader.uniform_float("color", (1.0, 0.15, 0.15, 1.0))
                batch_for_shader(self.shader, 'LINES',
                                 {"pos": self._seg(pts)}).draw(self.shader)

        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
      except Exception as ex:
        print("MK draw error:", ex)


# ----------------------------------------------------------------------------
#  LIVE PREVIEW (manual mode only) - single, reload-safe handler
# ----------------------------------------------------------------------------







_DNS_KEY = "mk_ls_preview_handle"
_GEN_KEY = "mk_ls_preview_gen"
_preview_handle = None
_preview_cache = {"key": None, "sec": None}


def _remove_preview_handler():
    global _preview_handle
    h = bpy.app.driver_namespace.get(_DNS_KEY)
    for handle in {h, _preview_handle}:
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            except Exception:
                pass
    bpy.app.driver_namespace[_DNS_KEY] = None
    _preview_handle = None


def _get_section_cached(obj):
    if obj is None:
        return None
    # match the real dovetail: if two cut halves are selected, use the shared
    # interface (mutual) so the preview matches curved / freehand caps too
    sel = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    other = None
    if len(sel) == 2 and obj in sel:
        other = sel[0] if sel[1] is obj else sel[1]
    key = (obj.name, len(obj.data.polygons),
           other.name if other else None,
           len(other.data.polygons) if other else 0)
    if _preview_cache["key"] != key:
        _preview_cache["key"] = key
        try:
            sec = None
            if other is not None:
                sec = find_section_mutual(obj, other)
                if sec is None:
                    sec = find_section_mutual(other, obj)
            if sec is None:
                sec = find_section(obj)
            _preview_cache["sec"] = sec
        except Exception:
            _preview_cache["sec"] = None
    return _preview_cache["sec"]


def _get_seam_loop_cached(obj):
    """Ordered world-space boundary loop of the cut cap, so the seam preview
    follows the section's real outline (circle stays circle, hex stays hex)
    instead of drawing a generic circle."""
    if obj is None:
        return None
    try:
        _mates = tuple(sorted(o.name for o in bpy.context.selected_objects
                              if o.type == 'MESH' and o is not obj))
    except Exception:
        _mates = ()
    key = (obj.name, len(obj.data.polygons), _mates, "seamloop")
    if _preview_cache.get("loop_key") != key:
        _preview_cache["loop_key"] = key
        _preview_cache["loop"] = None
        try:
            # same cap the operator will use, so the guide also shows up on a
            # curved freehand cut instead of only on flat ones
            mate = None
            try:
                others = [o for o in bpy.context.selected_objects
                          if o.type == 'MESH' and o is not obj]
                if others:
                    mate = others[0]
            except Exception:
                pass
            sec = _pick_shared_cap(obj, mate)
            if sec is None:
                return None
            center_w, normal_w, radius_w, face_idx = sec
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            faces = [bm.faces[i] for i in face_idx if i < len(bm.faces)]
            loop = _ordered_boundary_loop(bm, faces)
            if not loop or len(loop) < 3:
                bm.free()
                return None
            mw = obj.matrix_world
            pts = [mw @ v.co for v in loop]
            bm.free()
            _preview_cache["loop"] = (pts, center_w, normal_w, radius_w)
        except Exception:
            _preview_cache["loop"] = None
    return _preview_cache.get("loop")


def _find_cut_cap(obj):
    """Cut cap that also works on dense meshes.

    find_section clusters strictly coplanar faces, which shatters on high-poly
    cuts (thousands of tiny tris whose normals differ by more than the 2.5 deg
    tolerance), so it can return only a sliver of the real cap. Here that
    result is just a seed: the region is grown across edges to every
    neighbouring face still sitting on the cut plane, so triangle density stops
    mattering. Returns (center_w, normal_w, radius_w, face_indices)."""
    base = find_section(obj, want_faces=True)
    if base is None:
        return None
    center_w, normal_w, radius_w, seed_idx = base
    try:
        mw = obj.matrix_world
        mwi = mw.inverted()
        n_l = (mwi.to_3x3() @ normal_w).normalized()
        c_l = mwi @ center_w
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        bb = [Vector(c) for c in obj.bound_box]
        diag = (Vector((max(v[i] for v in bb) for i in range(3)))
                - Vector((min(v[i] for v in bb) for i in range(3)))).length
        eps = max(diag * 0.002, 1e-7)

        region = set(i for i in seed_idx if i < len(bm.faces))
        stack = list(region)
        while stack:
            fi = stack.pop()
            for e in bm.faces[fi].edges:
                for lf in e.link_faces:
                    li = lf.index
                    if li in region:
                        continue
                    cc = lf.calc_center_median()
                    if abs((cc - c_l).dot(n_l)) <= eps \
                            and abs(lf.normal.dot(n_l)) >= 0.7:
                        region.add(li)
                        stack.append(li)

        faces = [bm.faces[i] for i in region]
        verts = set(v for f in faces for v in f.verts)
        cen = Vector((0.0, 0.0, 0.0))
        tot = 0.0
        for f in faces:
            a = f.calc_area()
            cen += f.calc_center_median() * a
            tot += a
        cen = (cen / tot) if tot > 0.0 else c_l
        new_center = mw @ cen
        rad = 0.0
        for v in verts:
            rad = max(rad, (mw @ v.co - new_center).length)
        idx = sorted(region)
        bm.free()
        if rad <= 0.0:
            return base
        return new_center, normal_w, rad, idx
    except Exception as ex:
        print("MK cut cap error:", ex)
        return base


def _cut_cap_candidates(obj, keep=8):
    """All flat cap candidates on obj, biggest first, as
    [(center_w, normal_w, radius_w, face_indices), ...].

    find_section only returns the LARGEST coplanar cluster, which is wrong for
    a piece that has been cut more than once: it then has several caps and the
    biggest one is usually not the interface with the half you selected."""
    out = []
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        cos_tol = 0.999
        visited = set()
        clusters = []
        for f0 in bm.faces:
            if f0 in visited or f0.normal.length < 1e-9:
                continue
            n0 = f0.normal.copy()
            cluster = []
            stack = [f0]
            visited.add(f0)
            while stack:
                f = stack.pop()
                cluster.append(f)
                for e in f.edges:
                    for lf in e.link_faces:
                        if lf in visited or lf.normal.length < 1e-9:
                            continue
                        if lf.normal.dot(n0) >= cos_tol:
                            visited.add(lf)
                            stack.append(lf)
            area = sum(f.calc_area() for f in cluster)
            clusters.append((area, cluster, n0))
        clusters.sort(key=lambda c: c[0], reverse=True)

        mw = obj.matrix_world
        for area, cluster, n0 in clusters[:keep]:
            cverts = set()
            tot = 0.0
            cen = Vector((0.0, 0.0, 0.0))
            for f in cluster:
                a = f.calc_area()
                cen += f.calc_center_median() * a
                tot += a
                cverts.update(f.verts)
            if tot <= 0.0:
                continue
            cen /= tot
            center_w = mw @ cen
            normal_w = (mw.to_3x3() @ n0).normalized()
            radius_w = 0.0
            for v in cverts:
                radius_w = max(radius_w, (mw @ v.co - center_w).length)
            if radius_w <= 0.0:
                continue
            out.append((center_w, normal_w, radius_w,
                        [f.index for f in cluster]))
        bm.free()
    except Exception as ex:
        print("MK cap candidates error:", ex)
    return out


def _pick_facing_cap(obj, mate_center_w, mate_normal_w, mate_radius=None):
    """The cap on obj that mates with the cap already chosen on the other half.

    Two things have to hold, and loosening either lets the wrong face win:
    the cap has to look back at the other one (within about 25 degrees), and
    it has to sit on top of it - no further away than the cap's own radius.
    Allowing 72 degrees and any distance let a face 11 units away, tilted 66
    degrees, take the job, so the socket was carved nowhere near the tab and
    the two halves would not go together."""
    best = None
    try:
        want = mate_normal_w.normalized()
        for cand in _cut_cap_candidates(obj, keep=12):
            c, n, r, idx = cand
            if n.normalized().dot(want) > -0.9:      # ~25 degrees
                continue
            d = (c - mate_center_w).length
            limit = (mate_radius if mate_radius else r)
            if d > max(limit, 1e-6):
                continue
            if best is None or d < best[0]:
                best = (d, cand)
    except Exception as ex:
        print("MK facing cap error:", ex)
    return best[1] if best is not None else None


def _pick_shared_cap(obj, mate):
    """The cap on obj that actually faces `mate` - the interface those two
    pieces share. Falls back to the plain largest cap."""
    cands = _cut_cap_candidates(obj)
    if not cands:
        return _find_cut_cap(obj)
    if mate is None or len(cands) == 1:
        return cands[0]
    try:
        from mathutils.bvhtree import BVHTree
        dg = bpy.context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(mate, dg)
        mwi = mate.matrix_world.inverted()
        best = None
        for cand in cands:
            loc, _n, _i, dist = bvh.find_nearest(mwi @ cand[0])
            if loc is None:
                continue
            if best is None or dist < best[0]:
                best = (dist, cand)
        if best is not None:
            return best[1]
    except Exception as ex:
        print("MK shared cap error:", ex)
    return cands[0]


def _ordered_boundary_loop(bm, faces):
    """Ordered vertex loop around a face region. Dense cuts often have vertices
    where more than two boundary edges meet; walking blindly there breaks the
    chain, so unvisited neighbours are preferred and the longest loop wins."""
    fset = set(faces)
    adj = {}
    for f in fset:
        for e in f.edges:
            if sum(1 for lf in e.link_faces if lf in fset) == 1:
                a, b = e.verts[0], e.verts[1]
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
    if not adj:
        return None
    best = None
    unused = set(adj.keys())
    while unused:
        start = unused.pop()
        loop = [start]
        seen = {start}
        prev = None
        cur = start
        while True:
            cands = [v for v in adj.get(cur, []) if v is not prev]
            if not cands:
                break
            nxt = None
            for v in cands:
                if v not in seen:
                    nxt = v
                    break
            if nxt is None:
                break              # only visited neighbours left: loop closed
            loop.append(nxt)
            seen.add(nxt)
            unused.discard(nxt)
            prev, cur = cur, nxt
            if len(loop) > 500000:
                break
        if best is None or len(loop) > len(best):
            best = loop
    return best


def _seam_boundary_world(obj, face_idx):
    """Ordered world-space boundary loop of the given face cluster."""
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        faces = [bm.faces[i] for i in face_idx if i < len(bm.faces)]
        loop = _ordered_boundary_loop(bm, faces)
        if not loop or len(loop) < 3:
            bm.free()
            return None
        mw = obj.matrix_world
        pts = [mw @ v.co for v in loop]
        bm.free()
        return pts
    except Exception as ex:
        print("MK seam boundary error:", ex)
        return None


def _dead_seam_boundary_world(obj, face_idx):
    """Ordered world-space boundary loop of the given face cluster."""
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        fset = set(i for i in face_idx if i < len(bm.faces))
        bedges = []
        for i in fset:
            for e in bm.faces[i].edges:
                cnt = sum(1 for lf in e.link_faces if lf.index in fset)
                if cnt == 1:
                    bedges.append(e)
        adj = {}
        for e in bedges:
            a, b = e.verts[0], e.verts[1]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        if not adj:
            bm.free()
            return None
        start = next(iter(adj))
        loop = [start]
        prev = None
        cur = start
        while True:
            nxts = [v for v in adj.get(cur, []) if v is not prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt is start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if len(loop) > 100000:
                break
        mw = obj.matrix_world
        pts = [mw @ v.co for v in loop]
        bm.free()
        return pts if len(pts) >= 3 else None
    except Exception as ex:
        print("MK seam boundary error:", ex)
        return None


def _seam_tab_loops(pts, center_w, normal_w, outward_w, inset_d, push_d,
                    taper_d):
    """Base + top rings of the tab, using the same maths as the preview and
    the operator, so all three agree exactly."""
    nrm = normal_w.normalized()
    base = []
    top = []
    for p in pts:
        rel = p - center_w
        axial = rel.dot(nrm) * nrm
        radial = rel - axial
        rl = radial.length
        if rl > 1e-9:
            nl = max(rl - inset_d, rl * 0.02)
            radial = radial * (nl / rl)
        b = center_w + axial + radial
        base.append(b)
        q = b + outward_w * push_d
        rel2 = q - center_w
        ax2 = rel2.dot(nrm) * nrm
        rad2 = rel2 - ax2
        rl2 = rad2.length
        if rl2 > 1e-9:
            nl2 = max(rl2 - taper_d, rl2 * 0.05)
            rad2 = rad2 * (nl2 / rl2)
        top.append(center_w + ax2 + rad2)
    return base, top


def _build_seam_solid(name, base, top, center_w, normal_w, outward_w,
                      grow, back):
    """Closed frustum matching the tab (grown by `grow` for fit clearance and
    extended `back` behind the cut plane) so it can be booleaned out of the
    opposite half to form the socket."""
    nrm = normal_w.normalized()

    def expand(p, amt):
        rel = p - center_w
        axial = rel.dot(nrm) * nrm
        radial = rel - axial
        rl = radial.length
        if rl > 1e-9:
            radial = radial * ((rl + amt) / rl)
        return center_w + axial + radial

    b2 = [expand(p, grow) - outward_w * back for p in base]
    t2 = [expand(p, grow) + outward_w * grow for p in top]
    verts = b2 + t2
    n = len(b2)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(n, 2 * n)))
    me = bpy.data.meshes.new(name)
    me.from_pydata([list(v) for v in verts], [], faces)
    me.validate(verbose=False)
    me.update()
    return bpy.data.objects.new(name, me)


def _preview_coords(center_w, normal_w, radius_w, scene, inflate=1.0):
    size_ref = max(radius_w * scene.mk_scale, 1e-5)
    sz = -1.0 if getattr(scene, "mk_flip", False) else 1.0
    quat = normal_w.to_track_quat('Z', 'Y')

    shape = getattr(scene, "mk_key_shape", 'DOVETAIL')
    if shape == 'BALL':
        return _ball_preview_coords(center_w, quat, size_ref, sz,
                                    h_factor=scene.mk_height,
                                    ball_inflate=inflate)
    size_ref = size_ref * inflate
    Wb = size_ref * DT_WIDTH
    H = size_ref * scene.mk_height
    D = size_ref * DT_DEPTH
    hh = H / 2.0

    if shape == 'CYLINDER' or shape == 'HEX' or shape == 'TAPERED':
        import math
        seg = 6 if shape == 'HEX' else 16
        r0 = Wb * 0.5
        r1 = r0 * (DT_TAPER if shape == 'TAPERED' else 1.0)
        coords = []
        for zi, rr in ((-hh * sz, r0), (hh * sz, r1)):
            ring = []
            for i in range(seg):
                a = 2.0 * math.pi * i / seg
                ring.append(center_w + (quat @ Vector((rr * math.cos(a),
                                                       rr * math.sin(a), zi))))
            for i in range(seg):
                coords.append(ring[i])
                coords.append(ring[(i + 1) % seg])
        for i in range(seg):
            a = 2.0 * math.pi * i / seg
            coords.append(center_w + (quat @ Vector((r0 * math.cos(a),
                                                     r0 * math.sin(a), -hh * sz))))
            coords.append(center_w + (quat @ Vector((r1 * math.cos(a),
                                                     r1 * math.sin(a), hh * sz))))
        return coords

    if shape == 'BOX':
        hb, db = Wb / 2.0, D / 2.0
        local = [
            (-hb, -db, -hh * sz), (hb, -db, -hh * sz),
            (hb, db, -hh * sz), (-hb, db, -hh * sz),
            (-hb, -db, hh * sz), (hb, -db, hh * sz),
            (hb, db, hh * sz), (-hb, db, hh * sz),
        ]
    elif shape == 'CROSS':
        w, L = Wb * 0.18, D * 0.5
        local = [
            (-L, -w, -hh * sz), (L, -w, -hh * sz), (L, w, -hh * sz), (-L, w, -hh * sz),
            (-L, -w, hh * sz), (L, -w, hh * sz), (L, w, hh * sz), (-L, w, hh * sz),
        ]
    elif shape == 'TSLOT':
        sw, sd, hw, hd = Wb * 0.22, D * 0.28, Wb * 0.5, D * 0.5
        local = [
            (-sw, -sd, -hh * sz), (sw, -sd, -hh * sz), (sw, sd, -hh * sz), (-sw, sd, -hh * sz),
            (-hw, -hd, hh * sz), (hw, -hd, hh * sz), (hw, hd, hh * sz), (-hw, hd, hh * sz),
        ]
    else:
        # DOVETAIL default
        Wt = Wb * DT_TAPER
        hb, ht = Wb / 2.0, Wt / 2.0
        db, dt = D / 2.0, (D * DT_TAPER) / 2.0
        local = [
            (-hb, -db, -hh * sz), (hb, -db, -hh * sz),
            (hb, db, -hh * sz), (-hb, db, -hh * sz),
            (-ht, -dt, hh * sz), (ht, -dt, hh * sz),
            (ht, dt, hh * sz), (-ht, dt, hh * sz),
        ]
    w = [center_w + (quat @ Vector(p)) for p in local]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    coords = []
    for a, b in edges:
        coords.append(w[a])
        coords.append(w[b])
    return coords


def _ball_preview_coords(center_w, quat, size_ref, sz, h_factor=1.0,
                         segments=20, ball_inflate=1.0):
    """Wireframe guide for the ball key: a neck line + rings for the ball.
    Height (h_factor) lengthens the neck, matching the real peg.
    ball_inflate>1 grows ONLY the ball radius (used to draw the socket hole as
    a uniform offset around the ball, not a scale of the whole key)."""
    import math
    ball_r = size_ref * 0.45 * ball_inflate
    neck_r = (size_ref * 0.45) * 0.55
    neck_len = (size_ref * 0.45) * 0.6
    ball_off = (size_ref * 0.45) * (0.9 + h_factor * 1.6)

    def W(x, y, z):
        return center_w + (quat @ Vector((x, y, z * sz)))

    coords = []

    # neck: 4 vertical lines around the stem, from base to ball
    for (ox, oy) in ((neck_r, 0), (-neck_r, 0), (0, neck_r), (0, -neck_r)):
        coords.append(W(ox, oy, -neck_len))
        coords.append(W(ox, oy, ball_off))

    # ball ring in X-Z plane
    prev = None
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        p = W(ball_r * math.cos(a), 0.0, ball_off + ball_r * math.sin(a))
        if prev is not None:
            coords.append(prev)
            coords.append(p)
        prev = p
    # ball ring in Y-Z plane
    prev = None
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        p = W(0.0, ball_r * math.cos(a), ball_off + ball_r * math.sin(a))
        if prev is not None:
            coords.append(prev)
            coords.append(p)
        prev = p
    # ball ring in X-Y plane (equator)
    prev = None
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        p = W(ball_r * math.cos(a), ball_r * math.sin(a), ball_off)
        if prev is not None:
            coords.append(prev)
            coords.append(p)
        prev = p

    return coords


def _seam_ring_coords(center_w, normal_w, radius_w, radial_scale, depth_frac,
                      segments=48, push_dir=None):
    """A ring lying in the cut plane at `radial_scale` * radius, pushed by
    depth_frac*radius along `push_dir` (defaults to +normal). Used to preview
    the tab: outer ring = seam edge, inner+raised ring = top of the tab. The
    push direction must match the real tab so the guide and result agree."""
    import math
    n = normal_w.normalized()
    up = Vector((0.0, 0.0, 1.0))
    if abs(n.dot(up)) > 0.95:
        up = Vector((0.0, 1.0, 0.0))
    u = n.cross(up).normalized()
    v = n.cross(u).normalized()
    r = radius_w * radial_scale
    pd = (push_dir if push_dir is not None else n).normalized()
    push = pd * (depth_frac * radius_w)
    pts = []
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        p = center_w + u * (r * math.cos(a)) + v * (r * math.sin(a)) + push
        pts.append(p)
    coords = []
    for i in range(len(pts) - 1):
        coords.append(pts[i])
        coords.append(pts[i + 1])
    return coords


def _draw_preview():
    try:
        # only the newest registered handler draws; stale ones are inert
        if bpy.app.driver_namespace.get(_GEN_KEY) != id(_draw_preview):
            return
        ctx = bpy.context
        scene = getattr(ctx, "scene", None)
        show_key = getattr(scene, "mk_manual", False)
        show_seam = (getattr(scene, "mk_seam_preview", False)
                     and not getattr(scene, "mk_freehand", False)
                     and not getattr(scene, "mk_pen", False))
        if scene is None or not (show_key or show_seam):
            return
        obj = ctx.view_layer.objects.active
        if obj is None or obj.type != 'MESH':
            return
        sec = _get_section_cached(obj)
        if sec is None:
            return
        center_w, normal_w, radius_w = sec

        name = 'UNIFORM_COLOR' if bpy.app.version >= (4, 0, 0) else '3D_UNIFORM_COLOR'
        shader = gpu.shader.from_builtin(name)

        prev_depth = 'NONE'
        try:
            prev_depth = gpu.state.depth_test_get()
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(False)
        except Exception:
            pass
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        shader.bind()

        if show_seam:
            # tab preview (orange only): the raised tab the button will make -
            # a base ring at the cut edge + a top ring inset by Offset and
            # raised by Depth, with wall lines so it reads as a 3D tab.
            depth_f = getattr(scene, "mk_seam_depth", 0.05)
            off_f = getattr(scene, "mk_seam_offset", 0.1)

            # Draw on the half the operator will actually build on, and work
            # out the direction the same way it does. Two mismatches used to
            # invert the guide against the result: the preview measured from
            # the vertex average while the operator uses the bounding-box
            # centre, and the preview negated on Flip while the operator swaps
            # the halves instead.
            seam_obj = obj
            try:
                sel_m = [o for o in ctx.selected_objects if o.type == 'MESH']
                if len(sel_m) == 2 and getattr(scene, "mk_flip", False):
                    seam_obj = sel_m[0] if sel_m[1] is obj else sel_m[1]
            except Exception:
                seam_obj = obj

            loop = _get_seam_loop_cached(seam_obj)
            if loop is None and seam_obj is not obj:
                seam_obj = obj
                loop = _get_seam_loop_cached(obj)

            s_normal = loop[2] if loop is not None else normal_w
            s_center = loop[1] if loop is not None else center_w
            outward = s_normal.normalized()
            base = None
            top = None
            if loop is not None:
                pts, lc, ln, lr = loop
                # resample to the same segment count the operator will use, so
                # the guide shows the tab that actually gets built
                segs_p = getattr(scene, "mk_seam_segments", 48)
                n_pts = len(pts)
                segs_p = max(6, min(int(segs_p), n_pts))
                ridx = sorted(set(int(round(k * n_pts / float(segs_p))) % n_pts
                                  for k in range(segs_p)))
                if len(ridx) >= 3:
                    pts = [pts[i] for i in ridx]
                nrm = ln.normalized()
                inset_d = off_f * lr          # same absolute inset the op uses
                push_d = depth_f * lr
                moved = []
                for p in pts:
                    rel = p - lc
                    axial = rel.dot(nrm) * nrm
                    radial = rel - axial
                    rl = radial.length
                    if rl > 1e-9:
                        # pull inward by a CONSTANT distance, matching
                        # inset_region's thickness (not a uniform scale), so
                        # the outline keeps the section's own shape
                        newlen = max(rl - inset_d, rl * 0.02)
                        radial = radial * (newlen / rl)
                    moved.append(lc + axial + radial)
                base = []
                top = []
                taper_d = push_d * SEAM_TAPER
                nseg = len(moved)

                def _taper(p):
                    q = p + outward * push_d
                    rel = q - lc
                    axial = rel.dot(nrm) * nrm
                    radial = rel - axial
                    rl = radial.length
                    if rl > 1e-9:
                        nl2 = max(rl - taper_d, rl * 0.05)
                        radial = radial * (nl2 / rl)
                    return lc + axial + radial

                for i in range(nseg):
                    a = moved[i]
                    b = moved[(i + 1) % nseg]
                    base.append(a)
                    base.append(b)
                    top.append(_taper(a))
                    top.append(_taper(b))
            if base is None:
                base = _seam_ring_coords(center_w, normal_w, radius_w,
                                         1.0 - off_f, 0.0)
                top = _seam_ring_coords(center_w, normal_w, radius_w,
                                        1.0 - off_f, depth_f,
                                        push_dir=outward)
            walls = []
            for i in range(0, len(base), 2):
                walls.append(base[i])
                walls.append(top[i])
            b_base = batch_for_shader(shader, 'LINES', {"pos": base})
            b_top = batch_for_shader(shader, 'LINES', {"pos": top})
            b_wall = batch_for_shader(shader, 'LINES', {"pos": walls})
            shader.uniform_float("color", (1.0, 0.55, 0.1, 1.0))
            b_base.draw(shader)
            b_top.draw(shader)
            b_wall.draw(shader)
            # Socket-gap guide (cyan): the tab outline grown outward by the
            # Socket Gap - i.e. the mouth of the mating pocket. Drawn nested
            # with the orange tab (same shape, same direction) so the fit
            # clearance reads at a glance.
            gap_f = getattr(scene, "mk_seam_gap", 0.05)
            if gap_f > 1e-6 and loop is not None and base:
                gap_d = gap_f * lr

                def _grow(p, amt):
                    rel = p - lc
                    axial = rel.dot(nrm) * nrm
                    radial = rel - axial
                    rl = radial.length
                    if rl > 1e-9:
                        radial = radial * ((rl + amt) / rl)
                    return lc + axial + radial

                s_mouth = [_grow(p, gap_d) for p in base]
                s_top = [_grow(p, gap_d) + outward * gap_d for p in top]
                s_wall = []
                for i in range(0, len(s_mouth), 2):
                    s_wall.append(s_mouth[i])
                    s_wall.append(s_top[i])
                shader.uniform_float("color", (0.15, 0.9, 1.0, 1.0))
                batch_for_shader(shader, 'LINES',
                                 {"pos": s_mouth}).draw(shader)
                batch_for_shader(shader, 'LINES',
                                 {"pos": s_top}).draw(shader)
                batch_for_shader(shader, 'LINES',
                                 {"pos": s_wall}).draw(shader)

        if show_key:
            # move the preview with the Move X / Move Y sliders too
            ckey = _key_offset_world(center_w, normal_w, radius_w, scene)
            coords = _preview_coords(ckey, normal_w, radius_w, scene)
            clr = getattr(scene, "mk_clearance", 0.03)
            socket_coords = _preview_coords(ckey, normal_w, radius_w, scene,
                                            inflate=1.0 + clr)
            batch = batch_for_shader(shader, 'LINES', {"pos": coords})
            socket_batch = batch_for_shader(shader, 'LINES',
                                            {"pos": socket_coords})
            shader.uniform_float("color", (0.15, 0.9, 1.0, 1.0))
            batch.draw(shader)
            shader.uniform_float("color", (1.0, 0.55, 0.1, 1.0))
            socket_batch.draw(shader)

        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
        try:
            gpu.state.depth_test_set(prev_depth)
        except Exception:
            pass
    except Exception as ex:
        print("MK preview error:", ex)


def _redraw_areas(context):
    scr = getattr(context, "screen", None)
    if scr:
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _toggle_update(self, context):
    _preview_cache["key"] = None
    _redraw_areas(context)


def _val_update(self, context):
    _redraw_areas(context)


def _freehand_update(self, context):
    """The two drawing styles are alternatives, so turning one on turns the
    other off rather than leaving both ticked."""
    if self.mk_freehand and getattr(self, "mk_pen", False):
        self.mk_pen = False
    _redraw_areas(context)


def _pen_update(self, context):
    if self.mk_pen and getattr(self, "mk_freehand", False):
        self.mk_freehand = False
    _redraw_areas(context)


# ----------------------------------------------------------------------------
#  CUT BACKUPS / UNDO
# ----------------------------------------------------------------------------

_BACKUP_COLL = "MK_Backups"


def _backup_collection(context):
    coll = bpy.data.collections.get(_BACKUP_COLL)
    if coll is None:
        coll = bpy.data.collections.new(_BACKUP_COLL)
    if coll.name not in context.scene.collection.children:
        try:
            context.scene.collection.children.link(coll)
        except Exception:
            pass
    coll.hide_viewport = True
    coll.hide_render = True
    return coll


def _create_backup(context, obj):
    try:
        coll = _backup_collection(context)
        existing = [o.get("mk_backup_index", -1) for o in coll.objects]
        idx = (max(existing) + 1) if existing else 0
        bk = obj.copy()
        bk.data = obj.data.copy()
        bk.name = "MK_Backup_%d_%s" % (idx, obj.name)
        coll.objects.link(bk)
        bk["mk_backup_index"] = idx
        bk["mk_orig_name"] = obj.name
        bk["mk_child_b"] = ""
        return bk
    except Exception as ex:
        print("MK backup error:", ex)
        return None


def _undo_last_cut(context):
    coll = bpy.data.collections.get(_BACKUP_COLL)
    if coll is None or not coll.objects:
        return False
    bk = max(coll.objects, key=lambda o: o.get("mk_backup_index", -1))
    orig_name = bk.get("mk_orig_name", "")
    child_b = bk.get("mk_child_b", "")
    # A Multi Cut can leave more than two pieces, so the backup carries a
    # list. Older backups only have mk_child_b; both are honoured, so an
    # undo still works on cuts made before 5.2.0.
    kids = [n for n in bk.get("mk_children", "").split(",") if n]

    # delete every piece this cut produced
    for nm in [orig_name, child_b] + kids:
        if nm:
            o = bpy.data.objects.get(nm)
            if o is not None and o is not bk:
                bpy.data.objects.remove(o, do_unlink=True)

    # restore the backup as the original object
    try:
        coll.objects.unlink(bk)
    except Exception:
        pass
    try:
        context.scene.collection.objects.link(bk)
    except Exception:
        pass
    for k in ("mk_backup_index", "mk_orig_name", "mk_child_b",
              "mk_children"):
        if k in bk.keys():
            del bk[k]
    if orig_name:
        bk.name = orig_name
    try:
        bk.hide_set(False)
    except Exception:
        pass
    bk.hide_render = False
    for o in list(context.selected_objects):
        o.select_set(False)
    bk.select_set(True)
    context.view_layer.objects.active = bk
    return True


# ---------------------------------------------------------------------------
#  MULTI CUT  (5.2.0)
#
#  A piece attached in more than one place cannot come off with one stroke.
#  On the goblin the forearm is held at the shoulder AND where the hand meets
#  the staff; a leg is held at the hip and again where the shin touches the
#  base. Worse than merely not separating, a single stroke there is REFUSED:
#  do_slice and do_surface_cut both accept a cut only once the mesh has
#  actually come apart, and neither stroke on its own ever does that.
#
#  Multi Cut stores the strokes instead of cutting on each one. Nothing
#  touches the model until Apply, and then every stored stroke goes onto the
#  same mesh BEFORE the shells are counted. The rule Milad trusts - "only
#  accept it if it really came apart" - is not weakened, it just moves from
#  each stroke to the whole batch.
#
#  Straight-line strokes are stored as the world-space slab and plane that
#  do_slice derives, so they survive orbiting between strokes, and they are
#  replayed with the same bisect + split_edges. do_slice itself is untouched.
#  Freehand and pen strokes are stored as their ribbon/disc surface object,
#  which is already world-space, and replayed as blade booleans.
# ---------------------------------------------------------------------------

_PENDING_COLL = "MK_PendingCuts"
_MK_PENDING = []
_MK_PENDING_OBJ = ""


def _pending_count():
    return len(_MK_PENDING)


def _pending_collection(context):
    coll = bpy.data.collections.get(_PENDING_COLL)
    if coll is None:
        coll = bpy.data.collections.new(_PENDING_COLL)
    if coll.name not in context.scene.collection.children:
        try:
            context.scene.collection.children.link(coll)
        except Exception:
            pass
    coll.hide_render = True
    return coll


def _to_pending_coll(context, o, name):
    """Park a stored cut's surface in its own collection, drawn as wire so it
    reads as a guide and never hides the model or gets picked by accident."""
    coll = _pending_collection(context)
    for c in list(o.users_collection):
        try:
            c.objects.unlink(o)
        except Exception:
            pass
    try:
        coll.objects.link(o)
    except Exception:
        pass
    o.name = name
    try:
        o.display_type = 'WIRE'
        o.show_in_front = True
        o.hide_render = True
        o.hide_select = True
    except Exception:
        pass
    return o


def _screen_pts_to_world(context, obj, pts):
    """Screen points dropped onto the plane through the object's centre that
    faces the viewer. Used only for the guide, never for the cut."""
    region = context.region
    rv3d = context.region_data
    dg = context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    mw = obj.matrix_world
    bbw = [mw @ Vector(c) for c in ev.bound_box]
    center = sum(bbw, Vector()) / 8.0
    out = []
    for p in pts:
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, p)
        d = view3d_utils.region_2d_to_vector_3d(region, rv3d, p)
        out.append(o + d * ((center - o).dot(d)))
    return out


def _ghost_pierce(context, obj, pts_screen):
    """The guide for a straight or freehand stroke, built to pass THROUGH the
    model at the place the stroke was actually drawn.

    5.2.3 dropped the stroke onto the plane through the object's bounding-box
    centre. On anything off to one side - an arm reaching out, a raised knee -
    that depth is nowhere near the limb, so the guide hung in mid air next to
    the part it was going to cut and told you nothing about where the cut
    would land. Here each point of the stroke is shot into the mesh and the
    guide is stretched between where that ray ENTERS and where it LEAVES, so
    it straddles the real geometry and sticks out either side by
    mk_guide_depth of the thickness it pierces."""
    try:
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None or len(pts_screen) < 2:
            return None
        mw = obj.matrix_world
        mwi = mw.inverted()
        bbw = [mw @ Vector(c) for c in obj.bound_box]
        centre = sum(bbw, Vector()) / 8.0
        diag = (Vector((max(v[i] for v in bbw) for i in range(3)))
                - Vector((min(v[i] for v in bbw) for i in range(3)))).length
        eps = max(diag * 2e-4, 1e-6)
        frac = max(getattr(context.scene, "mk_guide_depth", 0.2), 0.0)

        # Sample ALONG the stroke, not just at its ends. A straight cut is
        # dragged from one side of a limb to the other, so BOTH endpoints sit
        # in empty space beside it - shooting only those two rays missed the
        # mesh every time and the guide fell back to the bounding-box centre,
        # which is exactly what left it hanging in mid air. The samples in
        # between are the ones that actually cross the part, and they are what
        # the guide gets built from.
        samples = []
        if len(pts_screen) == 2:
            a2 = Vector((pts_screen[0][0], pts_screen[0][1]))
            b2 = Vector((pts_screen[1][0], pts_screen[1][1]))
            steps = 32
            for k in range(steps + 1):
                samples.append(a2 + (b2 - a2) * (k / float(steps)))
        else:
            samples = [Vector((p[0], p[1])) for p in pts_screen]

        rays, spans = [], []
        for p in samples:
            o = view3d_utils.region_2d_to_origin_3d(region, rv3d, p)
            d = view3d_utils.region_2d_to_vector_3d(region, rv3d, p)
            d.normalize()
            rays.append((o, d))
            hits = []
            t = 0.0
            for _ in range(16):
                ok, loc, _n, _i = obj.ray_cast(mwi @ (o + d * t),
                                               (mwi.to_3x3() @ d).normalized())
                if not ok:
                    break
                tw = ((mw @ loc) - o).dot(d)
                if tw <= t:
                    break
                hits.append(tw)
                t = tw + eps
            if hits:
                t0, t1 = min(hits), max(hits)
                if t1 - t0 < eps:
                    t1 = t0 + diag * 0.02
                spans.append((t0, t1))
            else:
                spans.append(None)

        have = [i for i, sp in enumerate(spans) if sp is not None]
        if not have:
            return None
        # A sample that missed borrows the depth of the nearest one that hit,
        # so the guide keeps the full width of the stroke instead of jumping
        # off to some unrelated depth at the ends.
        for i, sp in enumerate(spans):
            if sp is None:
                j = min(have, key=lambda k: abs(k - i))
                spans[i] = spans[j]

        near, far = [], []
        for (o, d), (t0, t1) in zip(rays, spans):
            m = (t1 - t0) * frac
            near.append(o + d * (t0 - m))
            far.append(o + d * (t1 + m))

        bmc = bmesh.new()
        vN = [bmc.verts.new(p) for p in near]
        vF = [bmc.verts.new(p) for p in far]
        for i in range(len(near) - 1):
            try:
                bmc.faces.new((vN[i], vN[i + 1], vF[i + 1], vF[i]))
            except ValueError:
                pass
        if not bmc.faces:
            bmc.free()
            return None
        bmesh.ops.recalc_face_normals(bmc, faces=bmc.faces[:])
        me = bpy.data.meshes.new("MK_PendingCut")
        bmc.to_mesh(me)
        bmc.free()
        ob = bpy.data.objects.new("MK_PendingCut", me)
        context.collection.objects.link(ob)
        return ob
    except Exception as ex:
        print("MK guide error:", ex)
        return None


def _ghost_line(context, pts_w, closed=False):
    """The on-screen guide for a stored cut: the same sheet shape, but only a
    slice of the depth.

    5.2.0/5.2.1 used the cutting sheet itself. That sheet is built to span the
    model front to back so the blade clears the silhouette, so on screen it
    hung right through the model like a curtain - with two or three cuts
    stored you could not see what to aim the next one at. Here the depth is a
    fraction of the STROKE's own length (mk_guide_depth), not of the model, so
    a small cut gets a small guide and a long one a longer one. What cuts is
    still the full-depth sheet; this is only what you look at."""
    try:
        if len(pts_w) < 2:
            return None
        pts = [Vector(p) for p in pts_w]
        lo = Vector((min(p[i] for p in pts) for i in range(3)))
        hi = Vector((max(p[i] for p in pts) for i in range(3)))
        span = (hi - lo).length
        if span < 1e-9:
            return None
        frac = getattr(context.scene, "mk_guide_depth", 0.6)
        half = span * max(frac, 0.01) * 0.5

        # Which way is "deep"? For a closed loop the sheet should sit along the
        # loop's own axis, otherwise it folds through itself. For an open line
        # the view direction is what the real blade uses too.
        vd = None
        if closed and len(pts) > 2:
            n = Vector((0.0, 0.0, 0.0))
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                n.x += (a.y - b.y) * (a.z + b.z)
                n.y += (a.z - b.z) * (a.x + b.x)
                n.z += (a.x - b.x) * (a.y + b.y)
            if n.length > 1e-9:
                vd = n.normalized()
        if vd is None:
            region = context.region
            rv3d = context.region_data
            vd = view3d_utils.region_2d_to_vector_3d(
                region, rv3d,
                (region.width / 2.0, region.height / 2.0)).normalized()

        bmc = bmesh.new()
        near = [bmc.verts.new(p - vd * half) for p in pts]
        far = [bmc.verts.new(p + vd * half) for p in pts]
        rng = list(range(len(pts) - 1))
        if closed and len(pts) > 2:
            rng.append(len(pts) - 1)
        for i in rng:
            j = (i + 1) % len(pts)
            try:
                bmc.faces.new((near[i], near[j], far[j], far[i]))
            except ValueError:
                pass
        if not bmc.faces:
            bmc.free()
            return None
        bmesh.ops.recalc_face_normals(bmc, faces=bmc.faces[:])
        me = bpy.data.meshes.new("MK_PendingCut")
        bmc.to_mesh(me)
        bmc.free()
        ob = bpy.data.objects.new("MK_PendingCut", me)
        context.collection.objects.link(ob)
        return ob
    except Exception as ex:
        print("MK guide error:", ex)
        return None


def _pending_add_line(context, obj, a, b):
    """Store a straight-line stroke. The screen line is turned into world
    space here, at draw time, so the stored cut stays put when the view is
    orbited to reach the next join."""
    global _MK_PENDING_OBJ
    region = context.region
    rv3d = context.region_data
    o1 = view3d_utils.region_2d_to_origin_3d(region, rv3d, a)
    d1 = view3d_utils.region_2d_to_vector_3d(region, rv3d, a)
    o2 = view3d_utils.region_2d_to_origin_3d(region, rv3d, b)
    d2 = view3d_utils.region_2d_to_vector_3d(region, rv3d, b)

    normal_w = d1.cross((o2 + d2) - o1)
    if normal_w.length < 1e-9:
        return False
    normal_w.normalize()

    nA_w = d1.cross(normal_w)
    if (o2 + d2 - o1).dot(nA_w) < 0:
        nA_w = -nA_w
    nB_w = d2.cross(normal_w)
    if (o1 + d1 - o2).dot(nB_w) < 0:
        nB_w = -nB_w

    # the guide is the drawn line itself, sitting at the model's own depth -
    # the cut replays the slab maths above, not this
    ghost = _ghost_pierce(context, obj, [a, b])
    if ghost is not None:
        _to_pending_coll(context, ghost, "MK_PendingCut")

    _MK_PENDING.append({
        "kind": 'LINE',
        "co": o1.copy(), "no": normal_w.copy(),
        "oA": o1.copy(), "nA": nA_w.copy(),
        "oB": o2.copy(), "nB": nB_w.copy(),
        "ghost": ghost.name if ghost is not None else "",
        "surf": "",
    })
    _MK_PENDING_OBJ = obj.name
    return True


def _pending_add_surf(context, obj, surf, closed=False, pts_w=None,
                      ghost=None):
    """Store a freehand or pen stroke. The cutting surface is already in world
    space so it is kept as-is until Apply, but it is hidden: what you see is
    the thin guide built from the stroke. Hidden with hide_set, never with
    hide_viewport - measured, hide_viewport drops the object out of the
    depsgraph and the solidified blade comes back as a single face."""
    global _MK_PENDING_OBJ
    if surf is None:
        return False
    _to_pending_coll(context, surf, "MK_PendingSurf")
    try:
        surf.hide_set(True)
    except Exception:
        pass
    if ghost is None and pts_w:
        ghost = _ghost_line(context, list(pts_w), closed)
    if ghost is not None:
        _to_pending_coll(context, ghost, "MK_PendingCut")
    _MK_PENDING.append({"kind": 'SURF', "surf": surf.name,
                        "ghost": ghost.name if ghost is not None else "",
                        "closed": bool(closed)})
    _MK_PENDING_OBJ = obj.name
    return True


def _pending_drop(p):
    for key in ("ghost", "surf"):
        nm = p.get(key, "")
        if nm:
            o = bpy.data.objects.get(nm)
            if o is not None:
                try:
                    bpy.data.objects.remove(o, do_unlink=True)
                except Exception:
                    pass


def _pending_pop(context):
    if not _MK_PENDING:
        return False
    _pending_drop(_MK_PENDING.pop())
    _redraw_areas(context)
    return True


def _pending_clear(context):
    while _MK_PENDING:
        _pending_drop(_MK_PENDING.pop())
    coll = bpy.data.collections.get(_PENDING_COLL)
    if coll is not None and not coll.objects:
        try:
            bpy.data.collections.remove(coll)
        except Exception:
            pass
    _redraw_areas(context)


def _bisect_pending_line(bm, obj, p):
    """Replay one stored straight-line cut on an open bmesh: the slab test,
    the bisect and the edge split are the same three steps do_slice takes.
    Kept separate so do_slice stays byte-identical to the 3.0.0 reference."""
    mw = obj.matrix_world
    mwi = mw.inverted()
    m3t = mw.to_3x3().transposed()

    co_l = mwi @ p["co"]
    no_l = (m3t @ p["no"]).normalized()
    oA_l = mwi @ p["oA"]
    nA_l = (m3t @ p["nA"]).normalized()
    oB_l = mwi @ p["oB"]
    nB_l = (m3t @ p["nB"]).normalized()

    diag = _bm_diag(bm)
    eps = -1e-4 * max(diag, 1e-6)

    def in_slab(q):
        return ((q - oA_l).dot(nA_l) >= eps) and ((q - oB_l).dot(nB_l) >= eps)

    slab_faces = [f for f in bm.faces if in_slab(f.calc_center_median())]
    if not slab_faces:
        return 0

    gverts, gedges = set(), set()
    for f in slab_faces:
        gverts.update(f.verts)
        gedges.update(f.edges)
    geom = list(gverts) + list(gedges) + list(slab_faces)

    res = bmesh.ops.bisect_plane(
        bm, geom=geom, dist=1e-5 * max(diag, 1e-6),
        plane_co=co_l, plane_no=no_l,
        clear_inner=False, clear_outer=False,
    )
    cut_edges = [e for e in res['geom_cut']
                 if isinstance(e, bmesh.types.BMEdge)]
    if not cut_edges:
        return 0
    bmesh.ops.split_edges(bm, edges=cut_edges)
    return len(cut_edges)


def _surf_plane(surf):
    """Average point and normal of a stored cutting surface, used only to
    give _weld_caps a band to work in."""
    smw = surf.matrix_world
    sverts = [smw @ v.co for v in surf.data.vertices]
    co = sum(sverts, Vector()) / max(len(sverts), 1)
    no = Vector((0.0, 0.0, 0.0))
    for poly in surf.data.polygons:
        no += (smw.to_3x3() @ poly.normal)
    if no.length < 1e-9:
        no = Vector((0.0, 0.0, 1.0))
    no.normalize()
    ax = max(range(3), key=lambda i: abs(no[i]))
    if no[ax] < 0.0:
        no = -no
    return co, no


def _weld_multi(objs, tol, planes):
    """Zip the blade gap shut across N pieces, welding each vertex ONCE.

    _weld_caps is written for a single pair and keeps its "already used" set
    inside the one call. Running it over every pair of an N-piece cut let the
    same cap vertex be pulled to a different midpoint by each pair it took
    part in, which collapsed neighbouring vertices onto each other and left
    the body non-manifold along the cut - measured as 161 bad edges on a
    three-piece cut. Here the claim is global: a vertex is welded once, by the
    first partner that matches it, and is off limits afterwards."""
    try:
        import numpy as np
        from mathutils.kdtree import KDTree
        world = {}
        for o in objs:
            me = o.data
            n = len(me.vertices)
            co = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get('co', co)
            M = o.matrix_world
            world[o.name] = (co.reshape(n, 3) @ np.array(M.to_3x3()).T
                             + np.array(M.translation))
        claimed = {o.name: set() for o in objs}
        band = tol * 4.0
        welded = 0
        for pco, pno in planes:
            pc = np.array(pco)
            pn = np.array(pno)
            caps = {}
            for o in objs:
                d = np.abs((world[o.name] - pc) @ pn)
                caps[o.name] = [int(i) for i in np.nonzero(d <= band)[0]]
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    a, b = objs[i], objs[j]
                    ia = [k for k in caps[a.name] if k not in claimed[a.name]]
                    ib = [k for k in caps[b.name] if k not in claimed[b.name]]
                    if not ia or not ib:
                        continue
                    kd = KDTree(len(ib))
                    for t, k in enumerate(ib):
                        kd.insert(Vector(world[b.name][k]), t)
                    kd.balance()
                    mai = a.matrix_world.inverted()
                    mbi = b.matrix_world.inverted()
                    va = a.data.vertices
                    vb = b.data.vertices
                    for k in ia:
                        wco = Vector(world[a.name][k])
                        loc, t, dist = kd.find(wco)
                        if dist is None or dist > tol:
                            continue
                        bk = ib[t]
                        if bk in claimed[b.name] or k in claimed[a.name]:
                            continue
                        mid = (wco + loc) * 0.5
                        va[k].co = mai @ mid
                        vb[bk].co = mbi @ mid
                        claimed[a.name].add(k)
                        claimed[b.name].add(bk)
                        world[a.name][k] = np.array(mid)
                        world[b.name][bk] = np.array(mid)
                        welded += 1
                    a.data.update()
                    b.data.update()
        print("MK multi weld: %d vert pairs zipped (each vertex once)"
              % welded)
    except Exception as ex:
        print("MK multi weld error:", ex)


_TRACE_NAME = "MK_JoinTrace"


def _remove_join_trace():
    o = bpy.data.objects.get(_TRACE_NAME)
    if o is not None:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass


def _join_trace(context, obj, pend):
    """Draw what is STILL holding a piece on, instead of leaving the user to
    guess.

    Apply can only report that nothing came apart; it cannot say where the
    piece is still joined, and hunting for that by eye on a crouching figure
    with limbs pressed against the body is the slow part. This applies the
    stored cuts to a THROWAWAY copy - the model is never touched - walks out
    from the 3D cursor, and keeps the trail back from the farthest face it can
    reach. The cuts have already split their own edges, so that walk cannot
    cross them: it can only leave through whatever still holds the piece, and
    the trail runs straight through it. Cut across the trail and the piece
    comes free. More than one join left means doing this again."""
    tmp = None
    bm = None
    try:
        me = _evaluated_mesh_copy(context, obj)
        tmp = bpy.data.objects.new("MK_Scratch", me)
        tmp.matrix_world = obj.matrix_world.copy()
        context.collection.objects.link(tmp)
        for p in pend:
            if p["kind"] != 'SURF':
                continue
            so = bpy.data.objects.get(p["surf"])
            if so is None:
                continue
            try:
                so.hide_viewport = False
                so.hide_set(False)
            except Exception:
                pass
            blade = _blade_from_surface(context, so, 0.00002 * 8.0)
            _apply_boolean(context, tmp, blade, 'DIFFERENCE', 'EXACT')
            try:
                bpy.data.objects.remove(blade, do_unlink=True)
            except Exception:
                pass
        bm = bmesh.new()
        bm.from_mesh(tmp.data)
        for p in pend:
            if p["kind"] == 'LINE':
                try:
                    _bisect_pending_line(bm, obj, p)
                except Exception as ex:
                    print("MK trace line error:", ex)
        if not bm.faces:
            return False, "Nothing to trace"

        mw = obj.matrix_world
        target = context.scene.cursor.location.copy()
        seed, bd = None, 1e18
        for f in bm.faces:
            d = ((mw @ f.calc_center_median()) - target).length
            if d < bd:
                bd, seed = d, f

        comp = _component(seed)
        if len(comp) < len(bm.faces):
            _remove_join_trace()
            return True, ("The piece at the 3D cursor is already free "
                          "(%d faces) - press Apply Cuts" % len(comp))

        from collections import deque
        parent = {seed: None}
        q = deque([seed])
        last = seed
        while q:
            f = q.popleft()
            last = f
            for e in f.edges:
                for lf in e.link_faces:
                    if lf not in parent:
                        parent[lf] = f
                        q.append(lf)
        path = []
        f = last
        while f is not None:
            path.append(mw @ f.calc_center_median())
            f = parent[f]
        path.reverse()
        if len(path) < 3:
            return False, "Could not trace the join"
        keep = path[:min(len(path), 70)]

        _remove_join_trace()
        bmc = bmesh.new()
        vs = [bmc.verts.new(p) for p in keep]
        for i in range(len(vs) - 1):
            try:
                bmc.edges.new((vs[i], vs[i + 1]))
            except ValueError:
                pass
        tme = bpy.data.meshes.new(_TRACE_NAME)
        bmc.to_mesh(tme)
        bmc.free()
        ob = bpy.data.objects.new(_TRACE_NAME, tme)
        context.collection.objects.link(ob)
        try:
            ob.display_type = 'WIRE'
            ob.show_in_front = True
            ob.hide_render = True
            ob.hide_select = True
        except Exception:
            pass
        e = keep[-1]
        return True, ("Still joined - MK_JoinTrace drawn. Cut across it, "
                      "e.g. near (%.1f, %.1f, %.1f)" % (e.x, e.y, e.z))
    except Exception as ex:
        print("MK join-trace error:", ex)
        import traceback
        traceback.print_exc()
        return False, "Could not trace the join"
    finally:
        if bm is not None:
            try:
                bm.free()
            except Exception:
                pass
        if tmp is not None:
            d = tmp.data
            try:
                bpy.data.objects.remove(tmp, do_unlink=True)
            except Exception:
                pass
            try:
                if d is not None and d.users == 0:
                    bpy.data.meshes.remove(d)
            except Exception:
                pass


def _shells_of(bm):
    shells = []
    seen = set()
    for f0 in bm.faces:
        if f0 in seen:
            continue
        compf = list(_component(f0))
        seen.update(compf)
        shells.append({"faces": compf, "n": len(compf)})
    return shells


def _commit_multi(context, obj, bm, shells, backup):
    """Split the cut mesh into every loose shell it now has. The largest
    stays in obj, the rest become _B, _C, _D ... Unlike _commit_cut this is
    not limited to two pieces, which is the whole point of Multi Cut."""
    src_me = obj.data
    mats = list(src_me.materials)
    base = src_me.name
    suffix = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    order = sorted(shells, key=lambda x: x["n"], reverse=True)

    pieces = []
    for k, sh in enumerate(order):
        nb = _extract(sh["faces"])
        _cap_and_clean(nb)
        tag = "_A" if k == 0 else "_" + suffix[(k - 1) % 25]
        me = bpy.data.meshes.new(base + tag)
        for m in mats:
            me.materials.append(m)
        nb.to_mesh(me)
        nb.free()
        pieces.append(me)
    bm.free()

    obj.modifiers.clear()
    stale = obj.data
    obj.data = pieces[0]
    obj.data.update()
    try:
        if stale.users == 0:
            bpy.data.meshes.remove(stale)
    except Exception:
        pass
    smooth_keep_sharp(context, obj)

    names = []
    for k in range(1, len(pieces)):
        ob = bpy.data.objects.new(obj.name + "_" + suffix[(k - 1) % 25],
                                  pieces[k])
        ob.matrix_world = obj.matrix_world.copy()
        context.collection.objects.link(ob)
        smooth_keep_sharp(context, ob)
        names.append(ob.name)

    if backup is not None:
        backup["mk_children"] = ",".join(names)
        if names:
            backup["mk_child_b"] = names[0]

    for o in list(context.selected_objects):
        o.select_set(False)
    obj.select_set(True)
    for n in names:
        o = bpy.data.objects.get(n)
        if o is not None:
            o.select_set(True)
    context.view_layer.objects.active = obj
    return names


def apply_pending_cuts(context, obj):
    """Put every stored stroke on the model, then count the shells once.

    Blades first, then the straight-line bisects, then the shell walk. The
    blade thickness and solver escalation is the one from do_surface_cut,
    but it now wraps the WHOLE batch: success means the batch came apart,
    which is the only thing that can be measured when no single stroke is
    expected to separate anything."""
    pend = list(_MK_PENDING)
    if not pend:
        return False, "No pending cuts"

    surfs = []
    for p in pend:
        if p["kind"] != 'SURF':
            continue
        so = bpy.data.objects.get(p["surf"])
        if so is not None and so.type == 'MESH' and len(so.data.polygons):
            # it was hidden while it was only a guide; the blade needs it
            # evaluated again
            try:
                so.hide_viewport = False
                so.hide_set(False)
            except Exception:
                pass
            surfs.append(so)
    lines = [p for p in pend if p["kind"] == 'LINE']
    if not surfs and not lines:
        return False, "The stored cuts are empty"

    src_me = _evaluated_mesh_copy(context, obj)
    pristine = src_me.copy()
    backup = _create_backup(context, obj)
    obj.modifiers.clear()
    obj.data = src_me
    obj.data.update()

    base_thick = 0.00002
    heavy = len(obj.data.polygons) > 250000
    if surfs:
        order = ('FAST', 'EXACT', 'MANIFOLD') if heavy \
            else ('EXACT', 'FAST', 'MANIFOLD')
        muls = (8.0, 40.0, 200.0) if heavy else (1.0, 8.0, 40.0, 200.0)
        attempts = [(base_thick * m, sv) for m in muls for sv in order]
    else:
        # nothing to boolean: the bisect path needs no escalation at all
        attempts = [(base_thick, 'NONE')]

    saved = None
    used_thickness = base_thick
    first = True
    for thick, solver in attempts:
        if not first:
            stale = obj.data
            obj.data = pristine.copy()
            obj.data.update()
            try:
                if stale.users == 0:
                    bpy.data.meshes.remove(stale)
            except Exception:
                pass
        first = False
        used_thickness = thick

        ok_all = True
        for so in surfs:
            blade = _blade_from_surface(context, so, thick)
            ok = _apply_boolean(context, obj, blade, 'DIFFERENCE', solver)
            try:
                bpy.data.objects.remove(blade, do_unlink=True)
            except Exception:
                pass
            if not ok or len(obj.data.polygons) == 0:
                ok_all = False
                break
        if not ok_all:
            continue

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        for p in lines:
            try:
                _bisect_pending_line(bm, obj, p)
            except Exception as ex:
                print("MK multi-cut line error:", ex)
        shells = _shells_of(bm)
        print("MK multi-cut: %d stroke(s), solver %s, thickness %.6f -> "
              "%d shell(s)" % (len(pend), solver, thick, len(shells)))
        if len(shells) >= 2:
            saved = (bm, shells)
            break
        bm.free()

    try:
        if pristine.users == 0:
            bpy.data.meshes.remove(pristine)
    except Exception:
        pass

    if saved is None:
        _undo_last_cut(context)
        return False, ("%d cut(s) applied but nothing came apart. Every "
                       "place the piece is still attached needs its own cut."
                       % len(pend))

    bm, shells = saved
    names = _commit_multi(context, obj, bm, shells, backup)

    if surfs:
        # zip the blade gap shut across all the pieces at once. One pass, with
        # a global claim per vertex - see _weld_multi for why per-pair welding
        # damaged the mesh on a three-piece cut.
        objs = [obj] + [bpy.data.objects.get(n) for n in names]
        objs = [o for o in objs if o is not None]
        planes = []
        for so in surfs:
            try:
                planes.append(_surf_plane(so))
            except Exception:
                continue
        if planes and len(objs) > 1:
            _weld_multi(objs, max(used_thickness, base_thick) * 6.0, planes)

    _pending_clear(context)
    msg = "Multi Cut: %d stroke(s) -> %d piece(s)" % (len(pend), len(shells))
    if len(shells) < len(pend) + 1:
        # Fewer pieces than strokes means at least one stroke separated
        # nothing. The batch test only asks whether ANYTHING came apart, so
        # without this the dead stroke passes in silence.
        msg += (" - fewer pieces than cuts, so %d cut(s) freed nothing. Any "
                "piece still joined needs a cut at every join."
                % (len(pend) + 1 - len(shells)))
    return True, msg


# ---------------------------------------------------------------------------
#  QUICK CUT - part detection
#
#  The detection, palette and preview are the pipeline from Smart Segmentation:
#  a decimated proxy, a Shape Diameter Function per face, then greedy merging
#  of neighbouring regions. Cutting by hand-placed planes and flood filling
#  between them never held together on a real body, because a plane has to
#  close a full ring before anything separates. Clustering has no such
#  requirement: every face belongs to exactly one part from the start.
# ---------------------------------------------------------------------------





















































class MK_OT_explode(bpy.types.Operator):
    """Slide the cut pieces straight out along their own cut plane, so every
joint can be seen at once.

Each piece travels along the normal of the face it was cut on, away from where
it was attached - the head lifts, the legs drop, the arms go out sideways. The
biggest piece stays put. Every piece remembers where it started, so the slider
always measures from that home position; Return puts them all back"""
    bl_idname = "mk.explode"
    bl_label = "Разнести детали"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        amount = getattr(scene, "mk_explode", 0.25)
        sel = [o for o in context.selected_objects if o.type == 'MESH']
        if len(sel) < 2:
            self.report({'ERROR'}, "Выделите хотя бы две детали")
            return {'CANCELLED'}

        for o in sel:
            if "mk_home" not in o.keys():
                o["mk_home"] = list(o.location)

        def home_of(o):
            return Vector(o["mk_home"])

        def centre_at_home(o):
            return _obj_center_world(o) - (o.location - home_of(o))

        # overall size with everything at home, so the same value reads the
        # same on any model
        pts = []
        for o in sel:
            shift = o.location - home_of(o)
            for c in o.bound_box:
                pts.append((o.matrix_world @ Vector(c)) - shift)
        lo = Vector((min(p[i] for p in pts) for i in range(3)))
        hi = Vector((max(p[i] for p in pts) for i in range(3)))
        size = (hi - lo).length or 1.0

        # Anchor = the piece that takes up the most space, not the one with
        # the most vertices. A hand is dense with fingers but tiny, and picking
        # it as the anchor sends the arm back toward the shoulder instead of
        # away from the body.
        def _bulk(o):
            bb = [Vector(c) for c in o.bound_box]
            d = (Vector((max(v[i] for v in bb) for i in range(3)))
                 - Vector((min(v[i] for v in bb) for i in range(3))))
            s = o.matrix_world.to_scale()
            return abs(d.x * s.x) * abs(d.y * s.y) * abs(d.z * s.z)

        big = max(sel, key=_bulk)
        big_c = centre_at_home(big)

        # Work out, with everything back at home, which cut face each piece is
        # joined on. A piece must slide away from ITS OWN neighbour, not away
        # from the body's centre: the arm hangs below the shoulder but well
        # above the middle of the torso, so "away from the body centre" sent it
        # up into the shoulder instead of out along the arm. Among the flat
        # faces that touch a neighbour the real cut section is the biggest one,
        # which also skips the little flat skin patches that touch by chance.
        from mathutils.bvhtree import BVHTree
        saved = {}
        for o in sel:
            saved[o] = o.location.copy()
            o.location = home_of(o)
        context.view_layer.update()
        dg = context.evaluated_depsgraph_get()
        trees = {}
        for o in sel:
            try:
                trees[o.name] = (BVHTree.FromObject(o, dg),
                                 o.matrix_world.inverted())
            except Exception:
                pass

        tol = size * 0.02
        joint_dir = {}
        touches = {o: set() for o in sel}
        for o in sel:
            best = None
            try:
                cands = _cut_cap_candidates(o)
            except Exception:
                cands = []
            for c, n, r, _idx in cands:
                near = 1e18
                who = None
                for q in sel:
                    if q is o or q.name not in trees:
                        continue
                    bvh, qmwi = trees[q.name]
                    loc, _nr, _fi, d = bvh.find_nearest(qmwi @ c)
                    if loc is not None and d < near:
                        near = d
                        who = q
                if near <= tol:
                    if who is not None:
                        touches[o].add(who)
                        touches[who].add(o)
                    if o is not big and (best is None or r > best[1]):
                        best = (n.normalized(), r)
            if best is not None:
                joint_dir[o] = -best[0]

        # How many joints out from the anchor each piece sits. Pieces further
        # down a chain have to travel further, otherwise a whole limb slides
        # out as one block and the joints between its own pieces never open.
        depth = {big: 0}
        queue = [big]
        while queue:
            cur = queue.pop(0)
            for nb in touches.get(cur, ()):
                if nb not in depth:
                    depth[nb] = depth[cur] + 1
                    queue.append(nb)

        for o in sel:
            o.location = saved[o]
        context.view_layer.update()

        moved = 0
        for o in sel:
            home = home_of(o)
            if o is big:
                o.location = home
                continue

            d = joint_dir.get(o)
            if d is None:
                # nothing found touching: fall back to straight away from the
                # anchor, along the cut axis if there is one
                away = centre_at_home(o) - big_c
                d = None
                try:
                    sec = find_section(o)
                    if sec is not None:
                        d = sec[1].normalized()
                        if away.length > size * 1e-4 and d.dot(away) < 0:
                            d = -d
                except Exception:
                    d = None
                if d is None:
                    d = away.normalized() if away.length > size * 1e-4 \
                        else Vector((0.0, 0.0, 1.0))

            o.location = home + d * (size * amount * depth.get(o, 1))
            moved += 1

        context.view_layer.update()
        self.report({'INFO'}, "Разнесено деталей: %d" % moved)
        return {'FINISHED'}


class MK_OT_explode_reset(bpy.types.Operator):
    """Put every exploded piece back where it started"""
    bl_idname = "mk.explode_reset"
    bl_label = "Собрать обратно"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sel = [o for o in context.selected_objects if o.type == 'MESH']
        if not sel:
            sel = [o for o in context.scene.objects
                   if o.type == 'MESH' and "mk_home" in o.keys()]
        n = 0
        for o in sel:
            if "mk_home" in o.keys():
                o.location = Vector(o["mk_home"])
                del o["mk_home"]
                n += 1
        context.view_layer.update()
        if not n:
            self.report({'INFO'}, "Нечего возвращать")
            return {'CANCELLED'}
        self.report({'INFO'}, "Возвращено деталей: %d" % n)
        return {'FINISHED'}


class MK_OT_undo_cut(bpy.types.Operator):
    bl_idname = "mk.undo_cut"
    bl_label = "Отменить последний рез"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _undo_last_cut(context):
            self.report({'INFO'}, "Нет реза для отмены")
            return {'CANCELLED'}
        return {'FINISHED'}


class MK_OT_clear_cuts(bpy.types.Operator):
    bl_idname = "mk.clear_cuts"
    bl_label = "Удалить все резы"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        n = 0
        while _undo_last_cut(context) and n < 10000:
            n += 1
        coll = bpy.data.collections.get(_BACKUP_COLL)
        if coll is not None and not coll.objects:
            try:
                bpy.data.collections.remove(coll)
            except Exception:
                pass
        self.report({'INFO'}, "Отменено резов: %d" % n)
        return {'FINISHED'}


class MK_OT_recessed_seam(bpy.types.Operator):
    """Build a tab on one half and the matching socket in the other, so the
    pieces key together.

Select both halves: the active one gets the tab, the other gets the socket.
Flip swaps them - use it if the tab lands on the wrong side or has no room.
Depth and Offset scale with the cut section, so a small section needs a
higher Depth before the tab is visible.
Once built the tab sits inside the socket, so the socket is hidden until you
move a half aside"""
    bl_idname = "mk.recessed_seam"
    bl_label = "Утопленный шов"
    bl_options = {'REGISTER', 'UNDO'}

    depth: bpy.props.FloatProperty(
        name="Глубина", default=0.0, min=0.0, max=1.0,
    )
    offset: bpy.props.FloatProperty(
        name="Offset", default=0.0, min=0.0, max=1.0,
    )

    def execute(self, context):
        scene = context.scene
        depth_frac = getattr(scene, "mk_seam_depth", self.depth)
        offset_frac = getattr(scene, "mk_seam_offset", self.offset)
        segs = getattr(scene, "mk_seam_segments", 48)
        clearance = getattr(scene, "mk_seam_gap", 0.03)

        if (getattr(scene, "mk_freehand", False)
                or getattr(scene, "mk_pen", False)):
            self.report({'WARNING'},
                        "Утопленный шов только для прямых резов")
            return {'CANCELLED'}

        halves = [o for o in context.selected_objects if o.type == 'MESH']
        if len(halves) < 1:
            self.report({'ERROR'}, "Выделите разрезанные детали")
            return {'CANCELLED'}

        # One half gets the tab (the active one), the opposite half gets the
        # matching socket carved out of it. With a single selection it just
        # makes the tab. Flip swaps which half is which.
        male = context.active_object
        if male not in halves:
            male = halves[0]
        female = None
        if len(halves) >= 2:
            female = halves[0] if male is halves[1] else halves[1]
        if getattr(scene, "mk_flip", False) and female is not None:
            male, female = female, male

        # Freehand pieces take their own route: cap from the tag written at
        # cut time, and TAB ONLY. Straight-line pieces keep the original
        # behaviour end to end - cap by flat cluster, tab plus socket.
        # pick the cap that faces the other half, not just the biggest one
        sec = _pick_shared_cap(male, female)
        if sec is None:
            self.report({'WARNING'}, "Грань реза не найдена")
            return {'CANCELLED'}
        center_w, normal_w, radius_w, face_idx = sec
        inset_d = max(offset_frac * radius_w, 1e-5)
        push_d = max(depth_frac * radius_w, 1e-5)

        # Outward for the tab is the cap's own normal - Blender already keeps
        # face normals pointing out of the mesh. The bounding-box-centre test
        # that used to sit here got the sign wrong whenever the cap was off to
        # one side, so both halves came out with the same direction and Flip
        # only changed which object was touched, which read as Flip doing
        # nothing.
        male_out = normal_w.normalized()

        self._tab_geo = None
        self._tab_info = "cap=%d rim=0 seg=0 faces=0" % len(face_idx)

        socket = False
        why = ""

        if not self._make_tab(male, center_w, normal_w, radius_w,
                              inset_d, push_d, face_idx, segs,
                              force_outward=male_out):
            self.report({'WARNING'}, "Сбой выступа: %s" % self._tab_info)
            return {'CANCELLED'}

        if female is None:
            why = "only 1 piece selected - the socket needs the other half"
        else:
            fsec = _pick_facing_cap(female, center_w, normal_w,
                                    radius_w)
            if fsec is None:
                fsec = _pick_shared_cap(female, male)
            if fsec is None:
                why = "no cut face found on '%s'" % female.name
            else:
                fc, fn, fr, fidx = fsec
                f_inset = max(offset_frac * fr, 1e-5)
                f_push = max(depth_frac * fr, 1e-5)
                # opposite of the tab so the pocket carves the mating way
                socket = self._make_tab(
                    female, fc, fn, fr, f_inset, f_push, fidx, segs,
                    pocket=True, clearance=clearance,
                    force_outward=-male_out)
                if not socket:
                    why = "socket build failed on '%s'" % female.name

        if socket:
            self.report({'INFO'}, "Выступ на «%s», гнездо на «%s»"
                        % (male.name, female.name))
        else:
            self.report({'WARNING'}, "Выступ на «%s» — БЕЗ ГНЕЗДА: %s"
                        % (male.name, why))
        return {'FINISHED'}

    def _cut_socket(self, context, female, base_pts, top_pts, center_w,
                    normal_w, outward_w, radius_w, clearance):
        """Boolean-subtract the tab shape (grown by the Socket Gap) out of the
        opposite half, producing the mating pocket."""
        solid = None
        try:
            grow = max(clearance * radius_w, radius_w * 0.002)
            back = radius_w * 0.05
            solid = _build_seam_solid(
                "MK_SeamSolid", base_pts, top_pts, center_w, normal_w,
                outward_w, grow, back)
            context.collection.objects.link(solid)
            context.view_layer.update()

            mod = female.modifiers.new("MK_Socket", 'BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.object = solid
            try:
                mod.solver = 'EXACT'
            except Exception:
                pass
            with context.temp_override(
                    object=female, active_object=female,
                    selected_objects=[female],
                    selected_editable_objects=[female]):
                bpy.ops.object.modifier_apply(modifier=mod.name)
            return True
        except Exception as ex:
            print("MK socket error:", ex)
            try:
                if female.modifiers.get("MK_Socket"):
                    female.modifiers.remove(female.modifiers["MK_Socket"])
            except Exception:
                pass
            return False
        finally:
            if solid is not None:
                try:
                    bpy.data.objects.remove(solid, do_unlink=True)
                except Exception:
                    pass

    def _make_tab(self, obj, center_w, normal_w, radius_w, inset_d, push_d,
                  face_idx=None, segments=48, pocket=False, clearance=0.0,
                  force_outward=None):
        import bmesh
        self._tab_geo = None
        mw = obj.matrix_world
        mwi = mw.inverted()
        n_l = (mwi.to_3x3() @ normal_w).normalized()
        c_l = mwi @ center_w

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Use the EXACT cluster faces find_section measured, so the tab is
        # built on the same cross-section the preview is sized from - no
        # separate band search that could grab a smaller/larger set.
        if face_idx:
            seam = [bm.faces[i] for i in face_idx if i < len(bm.faces)]
        else:
            band = radius_w * 0.5
            seam = []
            for f in bm.faces:
                cc = f.calc_center_median()
                dp = abs((cc - c_l).dot(n_l))
                radial = ((cc - c_l) - (cc - c_l).dot(n_l) * n_l).length
                if dp <= band and abs(f.normal.dot(n_l)) >= 0.6 \
                        and radial <= radius_w * 1.5:
                    seam.append(f)
        if not seam:
            bm.free()
            return False

        # outward direction: from the body centre toward the cap
        body_c = Vector((0.0, 0.0, 0.0))
        for v in bm.verts:
            body_c += v.co
        body_c /= max(len(bm.verts), 1)
        cap_c = Vector((0.0, 0.0, 0.0))
        for f in seam:
            cap_c += f.calc_center_median()
        cap_c /= len(seam)
        if force_outward is not None:
            outward = (mwi.to_3x3() @ force_outward).normalized()
        else:
            outward = n_l if (cap_c - body_c).dot(n_l) >= 0 else -n_l

        # Offset/Depth arrive as WORLD distances, but every vert edit below is
        # in the object's LOCAL space. On a scaled object (this model is at
        # 0.004) that mismatch shrinks the tab by the scale factor until it is
        # invisible. Re-derive both from the cap's own local radius instead.
        cap_verts = set(v for f in seam for v in f.verts)
        radius_l = 0.0
        for v in cap_verts:
            rel = v.co - c_l
            radius_l = max(radius_l, (rel - rel.dot(n_l) * n_l).length)
        if radius_l > 1e-12 and radius_w > 1e-12:
            k = radius_l / radius_w
            inset_d = inset_d * k
            push_d = push_d * k
            clearance_l = clearance * radius_w * k
        else:
            clearance_l = clearance * radius_w

        # A socket carves INTO its own half (opposite its own outward), so when
        # the two halves meet the protruding tab drops into it. Widened by the
        # Socket Gap for fit.
        dir_vec = -outward if pocket else outward
        gap = clearance_l if pocket else 0.0

        # Keep the wall slope tied to the ORIGINAL depth so the pocket's cone
        # matches the tab's, then deepen by the gap. Without this the clearance
        # only existed sideways and the tab bottomed out on the pocket floor.
        taper_d = push_d * SEAM_TAPER
        if pocket:
            push_d = push_d + gap


        # Build the tab by REBUILDING the cap rather than inset+extrude. On a
        # dense cut the inset distance dwarfs a single face, so inset_region
        # folds the tiny faces into a tangled band. Building rings straight
        # from the boundary loop is immune to mesh density.
        loop = _ordered_boundary_loop(bm, seam)
        if not loop or len(loop) < 3:
            bm.free()
            return False

        def _shrink(p, amt, extra=None):
            q = p if extra is None else p + extra
            rel = q - c_l
            axial = rel.dot(n_l) * n_l
            radial = rel - axial
            rl = radial.length
            if rl > 1e-9:
                radial = radial * (max(rl - amt, rl * 0.05) / rl)
            return c_l + axial + radial

        def _grow(p, amt):
            rel = p - c_l
            axial = rel.dot(n_l) * n_l
            radial = rel - axial
            rl = radial.length
            if rl > 1e-9:
                radial = radial * ((rl + amt) / rl)
            return c_l + axial + radial

        rim = [v.co.copy() for v in loop]
        n_rim = len(rim)

        # Resample the tab rings. A dense cut gives a boundary loop with
        # thousands of points; carrying all of them into the tab makes a
        # needle-thin, unusable wall (and a nightmare for booleans). The rings
        # are rebuilt on `segments` evenly spaced points instead, while the
        # flat border still bridges back to every original rim vertex so the
        # body stays watertight.
        segs = max(6, min(int(segments), n_rim))
        idxs = sorted(set(int(round(k * n_rim / float(segs))) % n_rim
                          for k in range(segs)))
        if len(idxs) < 3:
            idxs = list(range(n_rim))
        segs = len(idxs)

        # For a socket the foot ring is widened by the Socket Gap so the tab
        # fits; the far ring is pushed INTO the body (dir_vec) instead of out.
        base = [_grow(_shrink(rim[i], inset_d), gap) for i in idxs]

        top = [_shrink(b, taper_d, extra=dir_vec * push_d) for b in base]

        # New geometry FIRST, old cap after: deleting the cap first can take
        # the boundary verts with it, which would break every face built from
        # them (and fail silently).
        base_v = [bm.verts.new(p) for p in base]
        top_v = [bm.verts.new(p) for p in top]
        bm.verts.index_update()

        import bisect

        def slot(i):
            k = bisect.bisect_right(idxs, i) - 1
            return k % segs

        made = 0
        # flat border: every rim edge bridges to the resampled foot ring
        for i in range(n_rim):
            j = (i + 1) % n_rim
            ka, kb = slot(i), slot(j)
            try:
                if ka == kb:
                    bm.faces.new((loop[i], loop[j], base_v[ka]))
                else:
                    bm.faces.new((loop[i], loop[j], base_v[kb], base_v[ka]))
                made += 1
            except Exception:
                pass
        # sloped wall (reverse winding for a pocket so normals face inward)
        for k in range(segs):
            k2 = (k + 1) % segs
            try:
                if pocket:
                    bm.faces.new((base_v[k2], base_v[k], top_v[k], top_v[k2]))
                else:
                    bm.faces.new((base_v[k], base_v[k2], top_v[k2], top_v[k]))
                made += 1
            except Exception:
                pass
        # far cap: fan from a centre vertex (tab top, or pocket floor)
        top_c = Vector((0.0, 0.0, 0.0))
        for p in top:
            top_c += p
        top_c /= max(len(top), 1)
        cv = bm.verts.new(top_c)
        for k in range(segs):
            k2 = (k + 1) % segs
            try:
                if pocket:
                    bm.faces.new((top_v[k2], top_v[k], cv))
                else:
                    bm.faces.new((top_v[k], top_v[k2], cv))
                made += 1
            except Exception:
                pass

        try:
            bmesh.ops.delete(bm, geom=seam, context='FACES')
        except Exception:
            pass

        self._tab_info = "cap=%d rim=%d seg=%d faces=%d%s" % (
            len(seam), n_rim, segs, made, " (pocket)" if pocket else "")
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        return made > 0

    def _dead_groove_one(self, obj, center_w, normal_w, radius_w, inset_d,
                         push_d, protrude=False, cap_idx=None):
        import bmesh
        mw = obj.matrix_world
        mwi = mw.inverted()
        n_l = (mwi.to_3x3() @ normal_w).normalized()
        c_l = mwi @ center_w

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        if cap_idx:
            # exact cap faces from the mutual face-to-face test
            seam_faces = [bm.faces[i] for i in cap_idx
                          if i < len(bm.faces)]
        else:
            # fallback: faces in a thin band on the seam plane that face along
            # the seam normal
            band = radius_w * 0.08
            near = []
            for f in bm.faces:
                c = f.calc_center_median()
                d = abs((c - c_l).dot(n_l))
                if d <= band:
                    near.append((d, f))
            seam_faces = [f for d, f in near
                          if abs(f.normal.dot(n_l)) >= 0.6]
            if len(seam_faces) < 1:
                seam_faces = [f for d, f in near]
        if not seam_faces:
            bm.free()
            return False

        # Reliable OUTWARD direction: from the body centre toward the cut cap.
        # (Averaging cap face normals is unreliable - a thin cut leaves faces
        # pointing both ways, cancelling out.)
        body_c = Vector((0.0, 0.0, 0.0))
        for v in bm.verts:
            body_c += v.co
        body_c /= max(len(bm.verts), 1)
        cap_c = Vector((0.0, 0.0, 0.0))
        nfc = 0
        for f in seam_faces:
            cap_c += f.calc_center_median()
            nfc += 1
        cap_c /= max(nfc, 1)
        outward = (cap_c - body_c)
        if outward.length < 1e-9:
            outward = n_l.copy()
        outward.normalize()
        # align with seam normal axis so it's clean
        if outward.dot(n_l) < 0:
            n_out = -n_l
        else:
            n_out = n_l
        outward = n_out

        # Step 1: inset by OFFSET to leave a flat outer border at the rim.
        if inset_d > 1e-5:
            bmesh.ops.inset_region(
                bm, faces=seam_faces, thickness=inset_d, depth=0.0,
                use_even_offset=True, use_boundary=True)

        # Step 2: inset again (taper width) for the sloped wall ring, then push
        # the floor along the seam normal: OUT for a male tab, IN for a female
        # pocket. The taper ring left behind makes the wall conical.
        taper = max(push_d * 0.6, 1e-5)
        bmesh.ops.inset_region(
            bm, faces=seam_faces, thickness=taper, depth=0.0,
            use_even_offset=True, use_boundary=True)
        # after the taper inset, seam_faces = the inner floor faces. Move ONLY
        # verts that belong entirely to the floor (all their faces are floor
        # faces), so the taper ring stays put and forms the sloped wall.
        floor_set = set(seam_faces)
        floor_verts = []
        seen_v = set()
        for f in seam_faces:
            for v in f.verts:
                if v in seen_v:
                    continue
                seen_v.add(v)
                if all((lf in floor_set) for lf in v.link_faces):
                    floor_verts.append(v)
        # fallback: if none qualify (tiny cap), move all floor verts
        if not floor_verts:
            floor_verts = list(seen_v)
        push_dir = outward if protrude else -outward
        for v in floor_verts:
            v.co = v.co + push_dir * push_d

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        return True


class MK_PT_line_slice(bpy.types.Panel):
    bl_label = "Шпонка"
    bl_idname = "MK_PT_line_slice"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Шпонка"

    def draw(self, context):
        layout = self.layout
        v = MK_VERSION
        layout.label(text="v%d.%d.%d" % v, icon='INFO')
        col = layout.column(align=True)
        col.label(text="1) Рез", icon='GREASEPENCIL')
        col.prop(context.scene, "mk_freehand")
        col.prop(context.scene, "mk_pen")
        col.prop(context.scene, "mk_multi")
        col.operator("mk.line_slice", text="Нарисовать линию реза", icon='GREASEPENCIL')

        if getattr(context.scene, "mk_multi", False):
            n = _pending_count()
            box = col.box()
            box.scale_y = 0.9
            box.label(text="Сохранено резов: %d" % n, icon='STICKY_UVS_LOC')
            row = box.row(align=True)
            row.enabled = n > 0
            row.operator("mk.apply_multi", text="Применить резы",
                         icon='CHECKMARK')
            row.operator("mk.clear_pending", text="Очистить", icon='X')
            box.prop(context.scene, "mk_guide_depth")
            box.operator("mk.find_join", text="Найти оставшееся соединение",
                         icon='VIEWZOOM')
            box.label(text="3D-курсор на детали, затем нажмите")
            if n == 0:
                box.label(text="По одному резу на стык, затем «Применить»")

        if getattr(context.scene, "mk_pen", False):
            col.prop(context.scene, "mk_pen_points")
            box = col.box()
            box.scale_y = 0.72
            box.label(text="Зажмите ЛКМ и рисуйте по модели")
            box.label(text="TAB — редактирование точек")
            box.label(text="TAB снова — обратно к рисованию")
            box.label(text="CTRL+клик по линии — добавить точку")
            box.label(text="DEL или X — удалить точку")
            box.label(text="под курсором")
            if getattr(context.scene, "mk_multi", False):
                box.label(text="ENTER — сохранить линию и очистить")
                box.label(text="перо для следующей; ENTER")
                box.label(text="на пустом пере — применить все")
            else:
                box.label(text="ENTER — рез, ESC — выход из пера")
        elif getattr(context.scene, "mk_freehand", False):
            col.label(text="Проведите штрих от руки через деталь")
        else:
            col.label(text="Тяните линию только через разрезаемую часть")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Утопленный шов (опционально)")
        seam_col = col.column(align=True)
        # The seam is built for the flat cross-section a straight cut leaves.
        # A freehand cut leaves a curved one, so the whole section is switched
        # off while Freehand is ticked.
        seam_col.enabled = not (getattr(context.scene, "mk_freehand", False)
                                or getattr(context.scene, "mk_pen", False))
        seam_col.prop(context.scene, "mk_seam_preview")
        seam_col.prop(context.scene, "mk_seam_depth")
        seam_col.prop(context.scene, "mk_seam_offset")
        seam_col.prop(context.scene, "mk_seam_segments")
        seam_col.prop(context.scene, "mk_seam_gap")
        seam_col.prop(context.scene, "mk_flip")
        seam_col.operator("mk.recessed_seam", text="Утопленный шов",
                          icon='MOD_BEVEL')
        if (getattr(context.scene, "mk_freehand", False)
                or getattr(context.scene, "mk_pen", False)):
            col.label(text="Только для прямых резов", icon='INFO')
        else:
            col.label(text="Для надёжного результата рисуйте", icon='INFO')
            col.label(text="линию реза всегда в одну сторону")
            col.label(text="(справа налево или слева направо)")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="2) Шпонка / ключ", icon='MESH_CUBE')
        col.prop(context.scene, "mk_key_shape", text="Форма")
        shape = getattr(context.scene, "mk_key_shape", 'DOVETAIL')
        if shape == 'BALL':
            col.label(text="Параметры шара — в окне оператора / Вручную", icon='INFO')
        elif shape in {'CYLINDER', 'HEX', 'TAPERED'}:
            col.label(text="Круглые/многогранные штифты — удобны для конечностей", icon='INFO')
        elif shape in {'BOX', 'CROSS', 'TSLOT'}:
            col.label(text="Антипроворот: коробка / крест / Т-шип", icon='INFO')
        col.prop(context.scene, "mk_manual")
        if context.scene.mk_manual:
            col.prop(context.scene, "mk_scale")
            col.prop(context.scene, "mk_height")
            col.prop(context.scene, "mk_flip")
            col.prop(context.scene, "mk_off_x")
            col.prop(context.scene, "mk_off_y")
            col.prop(context.scene, "mk_clearance")
        op = col.operator("mk.section_dovetail", text="Сделать ключ",
                          icon='MESH_CUBE')
        op.key_shape = context.scene.mk_key_shape
        col.label(text="Выделите обе половины, затем нажмите")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Разнести")
        col.prop(context.scene, "mk_explode")
        row = col.row(align=True)
        row.operator("mk.explode", text="Разнести", icon='MOD_PARTICLES')
        row.operator("mk.explode_reset", text="Вернуть", icon='LOOP_BACK')
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Отмена")
        col.operator("mk.undo_cut", text="Отменить последний рез", icon='LOOP_BACK')
        col.operator("mk.clear_cuts", text="Удалить все резы", icon='TRASH')
        layout.separator()

        # All the guidance lives here, at the end, so it never crowds the
        # controls. Fold it away once it is familiar.
        head = layout.row()
        head.prop(context.scene, "mk_show_tips",
                  icon='TRIA_DOWN' if context.scene.mk_show_tips
                  else 'TRIA_RIGHT',
                  emboss=False, text="Подсказки")
        if context.scene.mk_show_tips:
            box = layout.box()
            box.scale_y = 0.75
            box.label(text="Рез", icon='GREASEPENCIL')
            box.label(text="Тяните линию только через ту часть,")
            box.label(text="которую нужно разрезать, насквозь.")
            box.label(text="«От руки» — свободный штрих, рез при отпускании.")
            box.separator()
            box.label(text="Перо по поверхности", icon='GREASEPENCIL')
            box.label(text="Зажмите ЛКМ — линия по поверхности модели.")
            box.label(text="MMB — орбита, продолжайте штрих вокруг руки.")
            box.label(text="BACKSPACE — отменить штрих; замкните у старта.")
            box.label(text="TAB — правки точек; CTRL+клик — добавить;")
            box.label(text="X — удалить; ENTER — рез; ESC — отмена.")
            box.separator()
            box.label(text="Утопленный шов", icon='MOD_BEVEL')
            box.label(text="Выделите обе половины: активная — выступ,")
            box.label(text="вторая — гнездо. «Перевернуть» меняет стороны.")
            box.separator()
            box.label(text="Шпонка / ключ", icon='MESH_CUBE')
            box.label(text="Выделите обе половины → «Сделать ключ».")
            box.label(text="Формы: хвост, шар, цилиндр, коробка,")
            box.label(text="шестигранник, конус, крест, Т-шип.")
            box.label(text="Если ключа нет — «Вручную» + масштаб/высота.")
        layout.separator()
        box = layout.box()
        box.label(text="Dovetail Key RU — на базе Milad Kambari")
        box.label(text="3DRedbox Studio · русская доработка")
        row = box.row(align=True)
        row.operator("wm.url_open", text="ArtStation").url = \
            "https://artstation.com/milad_kambari"
        row.operator("wm.url_open", text="Superhive").url = \
            "https://superhivemarket.com/creators/miladkambari"
        row.operator("wm.url_open", text="Discord").url = \
            "https://discord.gg/EC8d7nPsYj"


class MK_OT_apply_multi(bpy.types.Operator):
    """Apply every stored Multi Cut stroke to the model at once.

Nothing is cut while the strokes are being drawn, so a piece attached in
more than one place - an arm held at the shoulder and again at the hand -
comes free the moment the last of its joins has a cut.
The result is accepted only if the model really came apart"""
    bl_idname = "mk.apply_multi"
    bl_label = "Применить резы"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _pending_count():
            self.report({'INFO'}, "Нет сохранённых резов")
            return {'CANCELLED'}
        obj = bpy.data.objects.get(_MK_PENDING_OBJ)
        if obj is None or obj.type != 'MESH':
            obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Объект, на котором рисовали резы, исчез")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        ok, msg = apply_pending_cuts(context, obj)
        self.report({'INFO'} if ok else {'WARNING'}, msg)
        _redraw_areas(context)
        return {'FINISHED'} if ok else {'CANCELLED'}


class MK_OT_find_join(bpy.types.Operator):
    """Draw where the piece is still attached.

Use it when Apply says nothing came apart. Put the 3D cursor on the piece you
want off, then click: the stored cuts are applied to a throwaway copy and the
trail still holding that piece on is drawn as MK_JoinTrace. Cut across the
trail anywhere outside your piece. If more than one join is left, cut this one
and click again"""
    bl_idname = "mk.find_join"
    bl_label = "Найти оставшееся соединение"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if not _pending_count():
            self.report({'INFO'}, "Нет сохранённых резов для трассировки")
            return {'CANCELLED'}
        obj = bpy.data.objects.get(_MK_PENDING_OBJ)
        if obj is None or obj.type != 'MESH':
            obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Выберите сетку, на которой рисовали резы")
            return {'CANCELLED'}
        ok, msg = _join_trace(context, obj, list(_MK_PENDING))
        self.report({'INFO'} if ok else {'WARNING'}, msg)
        _redraw_areas(context)
        return {'FINISHED'} if ok else {'CANCELLED'}


class MK_OT_clear_pending(bpy.types.Operator):
    """Throw away the stored Multi Cut strokes without cutting anything"""
    bl_idname = "mk.clear_pending"
    bl_label = "Очистить"
    bl_options = {'REGISTER'}

    def execute(self, context):
        n = _pending_count()
        _pending_clear(context)
        self.report({'INFO'}, "Очищено сохранённых резов: %d" % n)
        return {'FINISHED'}


classes = (MK_OT_line_slice, MK_OT_section_dovetail, MK_OT_recessed_seam,
           MK_OT_explode, MK_OT_explode_reset,
           MK_OT_undo_cut, MK_OT_clear_cuts,
           MK_OT_apply_multi, MK_OT_clear_pending, MK_OT_find_join,
           MK_PT_line_slice)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mk_freehand = bpy.props.BoolProperty(
        name="От руки",
        description="Draw a loose freehand stroke instead of a straight line. "
                    "The cut is made as soon as the mouse is released",
        default=False, update=_freehand_update,
    )
    bpy.types.Scene.mk_key_shape = bpy.props.EnumProperty(
        name="Форма ключа",
        description="Профиль шпонки / ключа",
        items=[
            ('DOVETAIL', "Ласточкин хвост", "Классический клиновой шип"),
            ('BALL', "Шар", "Шар на шейке (как замочная скважина)"),
            ('CYLINDER', "Цилиндр", "Круглый штифт"),
            ('BOX', "Коробка", "Квадратный штифт против проворота"),
            ('HEX', "Шестигранник", "Шестигранный штифт"),
            ('TAPERED', "Конус", "Самоцентрирующийся усечённый конус"),
            ('CROSS', "Крест", "Крестообразный ключ против проворота"),
            ('TSLOT', "Т-шип", "Т-образный шип (скользит и держит)"),
        ],
        default='DOVETAIL', update=_val_update,
    )
    bpy.types.Scene.mk_manual = bpy.props.BoolProperty(
        name="Вручную",
        description="Задать масштаб и высоту вручную с живым превью",
        default=False, update=_toggle_update,
    )
    bpy.types.Scene.mk_scale = bpy.props.FloatProperty(
        name="Масштаб",
        description="Множитель размера относительно авто-сечения",
        default=1.0, min=0.01, max=20.0, update=_val_update,
    )
    bpy.types.Scene.mk_height = bpy.props.FloatProperty(
        name="Высота",
        description="Коэффициент высоты шипа (вдоль нормали реза)",
        default=DT_HEIGHT, min=0.05, max=5.0, update=_val_update,
    )
    bpy.types.Scene.mk_flip = bpy.props.BoolProperty(
        name="Перевернуть",
        description="Перевернуть конусность вдоль нормали реза",
        default=False, update=_val_update,
    )
    bpy.types.Scene.mk_off_x = bpy.props.FloatProperty(
        name="Сдвиг X",
        description="Slide the key across the cut face (horizontal), as a "
                    "fraction of the cross-section radius",
        default=0.0, min=-3.0, max=3.0, update=_val_update,
    )
    bpy.types.Scene.mk_off_y = bpy.props.FloatProperty(
        name="Сдвиг Y",
        description="Slide the key across the cut face (vertical), as a "
                    "fraction of the cross-section radius",
        default=0.0, min=-3.0, max=3.0, update=_val_update,
    )
    bpy.types.Scene.mk_seam_preview = bpy.props.BoolProperty(
        name="Превью",
        description="Show the recessed-seam groove as an X-ray guide before "
                    "applying it",
        default=False, update=_val_update,
    )
    bpy.types.Scene.mk_seam_segments = bpy.props.IntProperty(
        name="Сегменты",
        description="How many segments the tab ring is rebuilt with. Dense "
                    "cuts have thousands of boundary points; resampling to a "
                    "sane count keeps the tab clean and boolean-friendly",
        default=48, min=6, max=512, update=_val_update,
    )
    bpy.types.Scene.mk_pen = bpy.props.BoolProperty(
        name="Перо по поверхности",
        description="Draw the cut line ON the model like a pen: paint, orbit "
                    "and carry on, edit the points, then press ENTER to cut. "
                    "Leave off to use the plain freehand stroke",
        default=False, update=_pen_update,
    )
    bpy.types.Scene.mk_multi = bpy.props.BoolProperty(
        name="Несколько резов",
        description="Collect several cuts and apply them together. A piece "
                    "held in more than one place - an arm joined at the "
                    "shoulder and again where the hand meets a staff - needs "
                    "one cut per join, and no single one of those cuts can "
                    "separate it on its own. Draw them all, then press Apply "
                    "Cuts. Работает с прямой линией, «От руки» и пером",
        default=False, update=_val_update,
    )
    bpy.types.Scene.mk_guide_depth = bpy.props.FloatProperty(
        name="Глубина направляющей",
        description="How far the guide for a stored cut sticks out past the "
                    "surface, as a fraction of the thickness it pierces. The "
                    "guide always spans the geometry under the stroke; this "
                    "is only the overhang, and only what you see - the cut "
                    "itself always spans the whole model",
        default=0.2, min=0.0, max=2.0, update=_val_update,
    )
    bpy.types.Scene.mk_pen_points = bpy.props.IntProperty(
        name="Точки",
        description="How many draggable points the painted freehand line is "
                    "reduced to. Fewer gives a smoother, easier line; more "
                    "follows the stroke closely",
        default=16, min=4, max=64, update=_val_update,
    )
    bpy.types.Scene.mk_explode = bpy.props.FloatProperty(
        name="Разнос",
        description="How far the pieces slide out from the main body, as a "
                    "fraction of the model size. 0 puts them back together",
        default=0.25, min=0.0, max=3.0,
    )
    bpy.types.Scene.mk_show_tips = bpy.props.BoolProperty(
        name="Подсказки",
        description="Показать подсказки внизу панели",
        default=True,
    )
    bpy.types.Scene.mk_seam_gap = bpy.props.FloatProperty(
        name="Зазор гнезда",
        description="How much wider the recessed socket is than the tab (fit "
                    "clearance). Larger = looser / easier fit",
        default=0.05, min=0.0, max=0.5, update=_val_update,
    )
    bpy.types.Scene.mk_seam_depth = bpy.props.FloatProperty(
        name="Глубина",
        description="Groove depth as a fraction of the cross-section radius",
        default=0.05, min=0.0, max=1.0,
    )
    bpy.types.Scene.mk_seam_offset = bpy.props.FloatProperty(
        name="Offset",
        description="How far the groove sits in from the seam edge, as a "
                    "fraction of the cross-section radius",
        default=0.1, min=0.0, max=1.0,
    )
    bpy.types.Scene.mk_clearance = bpy.props.FloatProperty(
        name="Зазор гнезда",
        description="How much bigger the socket hole is than the peg (fit "
                    "gap). Larger = looser/easier fit, the hole in side B "
                    "grows; smaller = tighter",
        default=0.03, min=0.0, max=0.5, update=_val_update,
    )
    global _preview_handle
    _remove_preview_handler()
    bpy.app.driver_namespace[_GEN_KEY] = id(_draw_preview)
    _preview_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_preview, (), 'WINDOW', 'POST_VIEW')
    bpy.app.driver_namespace[_DNS_KEY] = _preview_handle


def unregister():
    _remove_preview_handler()
    for attr in ("mk_freehand", "mk_key_shape", "mk_manual", "mk_scale", "mk_height", "mk_flip", "mk_off_x", "mk_off_y", "mk_clearance", "mk_seam_depth", "mk_seam_offset", "mk_seam_preview", "mk_seam_segments", "mk_seam_gap", "mk_show_tips", "mk_explode", "mk_pen_points", "mk_pen", "mk_multi", "mk_guide_depth"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
