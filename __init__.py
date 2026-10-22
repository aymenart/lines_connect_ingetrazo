# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# LINES CONNECT — Professional CAD Edge & Gap Extension for IngeTrazo
# Developed by: Aymen Alhaddad (Architect & Developer)
# Copyright (c) 2026 Aymen Alhaddad. All rights reserved.
# =============================================================================
"""Lines Connect — Advanced 3D Edge Connection and Extension for IngeTrazo.

Features:
- Connect Edges: Bridges the gap between edges. Intelligently connects to the body
  of a transverse edge (T-junction) at the exact intersection point and splits
  the boundary line without overlapping lines.
- Extend to Corner (Fillet R=0): Calculates the exact 3D intersection.
  * T-Junction: Extends the incoming line to hit the boundary line in front of it
    and STOPS, cleanly splitting the boundary line at the intersection.
  * L-Corner: Extends both lines outward to meet at a single corner vertex.
  * X-Crossing: Splits crossing lines at their intersection point.
- Close Gaps: Batch closes open endpoints and small gaps across multiple selected edges.
- Arc to Circle: Completes an arc along its natural curvature into a full 360-degree circle.

Interactive Picking:
- Real-time hover detection and rubber-band preview.
- Native IngeTrazo selection highlighting upon first click.
- Intelligent T-junction edge splitting to guarantee zero line overlap.
- Click-aware endpoint and boundary detection.
- Full transactional undo/redo integration with IngeTrazo.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Any

from PySide6.QtCore import Qt, QPointF, QSize, QTimer
from PySide6.QtGui import (
    QVector3D,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QIcon,
    QAction,
)
from PySide6.QtWidgets import (
    QApplication,
    QToolBar,
    QMenu,
)

from tools.base import Tool

KEY = "lines_connect"
VERSION = "1.3.0"
AUTHOR = "Aymen Alhaddad (Architect & Developer)"
TITLE = "Lines Connect"

TOLERANCE = 1e-4
PARALLEL_EPSILON = 1e-4
COPLANAR_EPSILON = 0.05  # 50 mm tolerance for 3D coplanar intersection

_GLOBAL_APP = None
_ACTIVE_TOOL = None


# =============================================================================
# 1. 3D VECTOR & GEOMETRIC COMPUTATION ENGINE
# =============================================================================

def to_v3d(v: Any) -> QVector3D:
    """Safely convert any vector or coordinate tuple to a PySide6 QVector3D."""
    if isinstance(v, QVector3D):
        return QVector3D(v.x(), v.y(), v.z())
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return QVector3D(float(v.x()), float(v.y()), float(v.z()))
    if isinstance(v, (tuple, list)) and len(v) >= 3:
        return QVector3D(float(v[0]), float(v[1]), float(v[2]))
    return QVector3D(v)


def distance_3d(p1: Any, p2: Any) -> float:
    """Euclidean distance between two 3D points."""
    v1 = to_v3d(p1)
    v2 = to_v3d(p2)
    return (v1 - v2).length()


def intersect_lines_3d_detailed(
    p1: QVector3D, v1: QVector3D, p2: QVector3D, v2: QVector3D, coplanar_tol: float = COPLANAR_EPSILON
) -> Tuple[bool, Optional[QVector3D], str, Tuple[float, float]]:
    """Calculate the exact 3D closest approach / intersection of two infinite lines.

    Lines are defined by:
        L1(t) = p1 + t * dir1
        L2(u) = p2 + u * dir2

    Returns:
        (success, intersection_point, status_message, (t, u))
    """
    v1_len = v1.length()
    v2_len = v2.length()
    if v1_len < 1e-7 or v2_len < 1e-7:
        return False, None, "Invalid line: zero length vector.", (0.0, 0.0)

    dir1 = v1 / v1_len
    dir2 = v2 / v2_len

    cross_dir = QVector3D.crossProduct(dir1, dir2)
    cross_len = cross_dir.length()
    delta_p = p2 - p1

    # Check for parallelism
    if cross_len < PARALLEL_EPSILON:
        collinear_dist = QVector3D.crossProduct(delta_p, dir1).length()
        if collinear_dist < 1e-3:
            return True, None, "collinear", (0.0, 0.0)
        return False, None, "Lines are parallel and do not intersect.", (0.0, 0.0)

    b = QVector3D.dotProduct(dir1, dir2)
    d = QVector3D.dotProduct(dir1, delta_p)
    e = QVector3D.dotProduct(dir2, delta_p)

    denom = 1.0 - (b * b)
    if abs(denom) < 1e-7:
        return False, None, "Lines are nearly parallel.", (0.0, 0.0)

    t = (d - b * e) / denom
    u = (b * d - e) / denom

    pt1 = p1 + dir1 * t
    pt2 = p2 + dir2 * u

    skew_distance = (pt2 - pt1).length()
    if skew_distance > coplanar_tol:
        return False, None, f"Lines are non-coplanar (skew distance: {skew_distance * 1000:.1f} mm).", (t, u)

    intersection_pt = (pt1 + pt2) * 0.5
    return True, intersection_pt, "Intersection found.", (t, u)


def compute_arc_circumcircle_3d(
    p1: Any, p2: Any, p3: Any
) -> Tuple[bool, Optional[QVector3D], Optional[QVector3D], float]:
    """Analytical 3D circumcircle through three ordered arc points (start, mid, end)."""
    v1 = to_v3d(p1)
    v2 = to_v3d(p2)
    v3 = to_v3d(p3)

    a = v1 - v3
    b = v2 - v3

    v_cross = QVector3D.crossProduct(a, b)
    v_len2 = v_cross.lengthSquared()
    if v_len2 < 1e-12:
        return False, None, None, 0.0

    normal = v_cross.normalized()

    # Exact closed-form circumcenter offset vector from p3
    numerator = QVector3D.crossProduct(
        b * a.lengthSquared() - a * b.lengthSquared(), v_cross
    )
    r_vec = numerator / (2.0 * v_len2)
    center = v3 + r_vec
    radius = (v1 - center).length()

    return True, center, normal, radius


# =============================================================================
# 2. SELECTION & TOPOLOGY HELPERS
# =============================================================================

def selected_edges(scene) -> List[Any]:
    """Retrieve currently selected edges from the active mesh."""
    try:
        from core.mesh import Edge
        live = set(scene.mesh.edges)
        return [e for e in scene.selection if isinstance(e, Edge) and e in live]
    except Exception:
        return [e for e in getattr(scene, "selection", []) if hasattr(e, "v0") and hasattr(e, "v1")]


def mark_mesh_modified(mesh, scene=None) -> None:
    """Mark the mesh and scene as dirty to ensure immediate viewport redraw."""
    if hasattr(mesh, "_chunk_dirty"):
        mesh._chunk_dirty = True
    if hasattr(mesh, "_mut_serial"):
        mesh._mut_serial += 1
    if scene and hasattr(scene, "version"):
        scene.version += 1


def pick_edge_at(viewport, x: float, y: float) -> Optional[Any]:
    """Safely pick a visible edge under screen coordinates (x, y)."""
    ix, iy = int(round(x)), int(round(y))
    pick = getattr(viewport, "pick_visible_edge", None) or getattr(viewport, "pick_edge", None)
    if callable(pick):
        try:
            edge = pick(ix, iy)
            from core.mesh import Edge
            if isinstance(edge, Edge):
                return edge
        except Exception:
            pass
    return None


def get_closest_endpoint(edge: Any, target_point: Any) -> Any:
    """Return the vertex of edge that is closest to target_point."""
    t_pt = to_v3d(target_point)
    d0 = (to_v3d(edge.v0.position) - t_pt).length()
    d1 = (to_v3d(edge.v1.position) - t_pt).length()
    return edge.v0 if d0 < d1 else edge.v1


def split_edge_safe(mesh, edge: Any, split_pt: QVector3D) -> Tuple[Optional[Any], Optional[Any]]:
    """Safely split an edge at split_pt into two edges (v0 -> split_pt) and (split_pt -> v1)."""
    p0 = to_v3d(edge.v0.position)
    p1 = to_v3d(edge.v1.position)
    s_pt = to_v3d(split_pt)

    if (p0 - s_pt).length() < 1e-4 or (p1 - s_pt).length() < 1e-4:
        return edge, None

    # 1. Use IngeTrazo's native mesh.split_edge if available
    if hasattr(mesh, "split_edge"):
        try:
            return mesh.split_edge(edge, s_pt)
        except Exception:
            pass

    # 2. Topological fallback
    curve_id = getattr(edge, "curve", None)
    if hasattr(mesh, "remove_edge"):
        try:
            mesh.remove_edge(edge)
        except Exception:
            pass

    e1 = mesh.add_edge(p0, s_pt)
    e2 = mesh.add_edge(s_pt, p1)

    if curve_id is not None:
        if hasattr(e1, "curve"):
            e1.curve = curve_id
        if hasattr(e2, "curve"):
            e2.curve = curve_id

    return e1, e2


def order_chain_vertices(edges: List[Any]) -> List[Any]:
    """Order vertices along a continuous unbranched chain from start to end."""
    if not edges:
        return []

    neighbors = {}
    for e in edges:
        neighbors.setdefault(e.v0, []).append((e.v1, e))
        neighbors.setdefault(e.v1, []).append((e.v0, e))

    endpoints = [v for v, n in neighbors.items() if len(n) == 1]
    start_v = endpoints[0] if endpoints else next(iter(neighbors))

    path = [start_v]
    visited_edges = set()
    curr = start_v

    while True:
        next_step = None
        for nxt_v, edge in neighbors.get(curr, []):
            if edge not in visited_edges:
                next_step = (nxt_v, edge)
                break
        if not next_step:
            break
        nxt_v, edge = next_step
        visited_edges.add(edge)
        path.append(nxt_v)
        curr = nxt_v

    return path


# =============================================================================
# 3. CORE CAD MESH MUTATIONS (T-JUNCTION AWARE CONNECT & EXTEND)
# =============================================================================

def op_extend_edges(
    mesh, edge1: Any, edge2: Any,
    click1: Optional[QVector3D] = None,
    click2: Optional[QVector3D] = None
) -> Tuple[bool, str]:
    """Extend lines to meet cleanly at intersection without overlapping lines.

    Handles:
    - T-Junction: Line 1 extends to Line 2 and STOPS, splitting Line 2 at intersection.
                  Or Line 2 extends to Line 1 and STOPS, splitting Line 1.
    - L-Corner: Both lines extend outward to meet at a single corner vertex.
    - X-Crossing: Both lines are split at their intersection point.
    """
    p_A0 = to_v3d(edge1.v0.position)
    p_A1 = to_v3d(edge1.v1.position)
    p_B0 = to_v3d(edge2.v0.position)
    p_B1 = to_v3d(edge2.v1.position)

    v1_full = p_A1 - p_A0
    v2_full = p_B1 - p_B0
    L1 = v1_full.length()
    L2 = v2_full.length()
    if L1 < 1e-6 or L2 < 1e-6:
        return False, "One of the selected edges has zero length."

    v1 = v1_full / L1
    v2 = v2_full / L2

    ok, int_pt, msg, (t, u) = intersect_lines_3d_detailed(p_A0, v1, p_B0, v2)
    if not ok:
        return False, msg

    if int_pt is None and msg == "collinear":
        return op_connect_edges(mesh, edge1, edge2, click1, click2)

    eps = 1e-3
    inside1 = (eps <= t <= L1 - eps)
    inside2 = (eps <= u <= L2 - eps)
    at_A = (abs(t) < eps or abs(t - L1) < eps)
    at_B = (abs(u) < eps or abs(u - L2) < eps)

    # -------------------------------------------------------------------------
    # Case 1: T-Junction: Edge 1 is incoming, Edge 2 is the boundary line
    # -------------------------------------------------------------------------
    if inside2 and not inside1:
        # Edge 2 is split at int_pt; Edge 2 is NOT extended!
        split_edge_safe(mesh, edge2, int_pt)
        # Edge 1 connects to int_pt and STOPS!
        if not at_A:
            tip1 = p_A1 if t > L1 else p_A0
            mesh.add_edge(tip1, int_pt)

    # -------------------------------------------------------------------------
    # Case 2: T-Junction: Edge 2 is incoming, Edge 1 is the boundary line
    # -------------------------------------------------------------------------
    elif inside1 and not inside2:
        # Edge 1 is split at int_pt; Edge 1 is NOT extended!
        split_edge_safe(mesh, edge1, int_pt)
        # Edge 2 connects to int_pt and STOPS!
        if not at_B:
            tip2 = p_B1 if u > L2 else p_B0
            mesh.add_edge(tip2, int_pt)

    # -------------------------------------------------------------------------
    # Case 3: X-Crossing: Both lines cross through each other's body
    # -------------------------------------------------------------------------
    elif inside1 and inside2:
        split_edge_safe(mesh, edge1, int_pt)
        split_edge_safe(mesh, edge2, int_pt)

    # -------------------------------------------------------------------------
    # Case 4: L-Corner: Neither edge contains int_pt; both extend to corner
    # -------------------------------------------------------------------------
    else:
        if not at_A:
            tip1 = p_A1 if t > L1 else p_A0
            mesh.add_edge(tip1, int_pt)
        if not at_B:
            tip2 = p_B1 if u > L2 else p_B0
            mesh.add_edge(tip2, int_pt)

    # -------------------------------------------------------------------------
    # Weld coincident vertices at intersection
    # -------------------------------------------------------------------------
    if hasattr(mesh, "weld_coincident"):
        try:
            mesh.weld_coincident()
        except Exception:
            pass

    mark_mesh_modified(mesh)
    return True, "Extended to intersection cleanly (zero line overlap)."


def op_connect_edges(
    mesh, edge1: Any, edge2: Any,
    click1: Optional[QVector3D] = None,
    click2: Optional[QVector3D] = None
) -> Tuple[bool, str]:
    """Connect Edge 1 and Edge 2.
    
    If one edge points into the body of the other (T-junction), connects
    directly to that intersection point and cleanly splits the boundary line.
    Otherwise bridges endpoints.
    """
    p_A0 = to_v3d(edge1.v0.position)
    p_A1 = to_v3d(edge1.v1.position)
    p_B0 = to_v3d(edge2.v0.position)
    p_B1 = to_v3d(edge2.v1.position)

    v1_full = p_A1 - p_A0
    v2_full = p_B1 - p_B0
    L1 = v1_full.length()
    L2 = v2_full.length()
    v1 = v1_full / L1 if L1 > 1e-6 else QVector3D(0, 0, 0)
    v2 = v2_full / L2 if L2 > 1e-6 else QVector3D(0, 0, 0)

    # Check for T-junction
    ok, int_pt, msg, (t, u) = intersect_lines_3d_detailed(p_A0, v1, p_B0, v2)
    if ok and int_pt is not None:
        eps = 1e-3
        inside1 = (eps <= t <= L1 - eps)
        inside2 = (eps <= u <= L2 - eps)
        at_A = (abs(t) < eps or abs(t - L1) < eps)
        at_B = (abs(u) < eps or abs(u - L2) < eps)

        # Case A: Edge 1 points at Edge 2's body
        if inside2 and not inside1:
            split_edge_safe(mesh, edge2, int_pt)
            if not at_A:
                tip1 = p_A1 if t > L1 else p_A0
                mesh.add_edge(tip1, int_pt)
            if hasattr(mesh, "weld_coincident"):
                try:
                    mesh.weld_coincident()
                except Exception:
                    pass
            mark_mesh_modified(mesh)
            return True, "Connected line to the line in front of it (T-junction)."

        # Case B: Edge 2 points at Edge 1's body
        elif inside1 and not inside2:
            split_edge_safe(mesh, edge1, int_pt)
            if not at_B:
                tip2 = p_B1 if u > L2 else p_B0
                mesh.add_edge(tip2, int_pt)
            if hasattr(mesh, "weld_coincident"):
                try:
                    mesh.weld_coincident()
                except Exception:
                    pass
            mark_mesh_modified(mesh)
            return True, "Connected line to the line in front of it (T-junction)."

    # Standard endpoint-to-endpoint bridge connection
    if click1 is not None and click2 is not None:
        v1_ep = get_closest_endpoint(edge1, click1)
        v2_ep = get_closest_endpoint(edge2, click2)
    else:
        pairs = [
            (edge1.v0, edge2.v0),
            (edge1.v0, edge2.v1),
            (edge1.v1, edge2.v0),
            (edge1.v1, edge2.v1),
        ]
        v1_ep, v2_ep = min(pairs, key=lambda pair: distance_3d(pair[0].position, pair[1].position))

    p1, p2 = to_v3d(v1_ep.position), to_v3d(v2_ep.position)
    dist = (p1 - p2).length()

    if dist < TOLERANCE:
        return False, "Endpoints are already touching."

    if hasattr(mesh, "find_edge") and mesh.find_edge(v1_ep, v2_ep) is not None:
        return False, "An edge already connects these two vertices."

    mesh.add_edge(p1, p2)
    if hasattr(mesh, "weld_coincident"):
        try:
            mesh.weld_coincident()
        except Exception:
            pass
    mark_mesh_modified(mesh)
    return True, f"Connected with a new edge ({dist * 1000:.1f} mm gap closed)."


def op_close_selected_gaps(mesh, edges: List[Any], max_gap: float = 0.05) -> Tuple[int, str]:
    """Find all open chain endpoints within selected edges and close gaps up to max_gap."""
    eset = set(edges)
    open_vertices = []

    for e in edges:
        for v in (e.v0, e.v1):
            selected_degree = sum(1 for edge in v.edges if edge in eset)
            if selected_degree == 1 and v not in open_vertices:
                open_vertices.append(v)

    if len(open_vertices) < 2:
        return 0, "No open endpoints found in the current selection."

    candidates = []
    for i in range(len(open_vertices)):
        vi = open_vertices[i]
        for j in range(i + 1, len(open_vertices)):
            vj = open_vertices[j]
            d = distance_3d(vi.position, vj.position)
            if d <= max_gap and d > TOLERANCE:
                candidates.append((d, vi, vj))

    candidates.sort(key=lambda t: t[0])
    used = set()
    closed_count = 0

    for dist, v1, v2 in candidates:
        if v1 in used or v2 in used:
            continue

        p1 = to_v3d(v1.position)
        p2 = to_v3d(v2.position)

        free1 = len(v1.edges) == 1 and not getattr(v1, "faces", [])
        free2 = len(v2.edges) == 1 and not getattr(v2, "faces", [])

        if free1 and free2:
            target = (p1 + p2) * 0.5
            mesh.place_vertex(v1, target)
            mesh.place_vertex(v2, target)
        elif free1:
            mesh.place_vertex(v1, p2)
        elif free2:
            mesh.place_vertex(v2, p1)
        else:
            mesh.add_edge(p1, p2)

        used.add(v1)
        used.add(v2)
        closed_count += 1

    if hasattr(mesh, "weld_coincident"):
        try:
            mesh.weld_coincident()
        except Exception:
            pass

    mark_mesh_modified(mesh)
    return closed_count, f"Closed {closed_count} open gap(s) successfully."


def op_arc_to_circle(mesh, seed_edge, all_selected_edges: List[Any] = None) -> Tuple[bool, str]:
    """Complete a selected circular arc seamlessly along its curvature into a full 360-degree circle."""
    curve_id = getattr(seed_edge, "curve", None)

    if curve_id is not None:
        arc_edges = [e for e in mesh.edges if getattr(e, "curve", None) == curve_id]
    elif all_selected_edges and len(all_selected_edges) >= 2:
        arc_edges = [e for e in all_selected_edges if e in set(mesh.edges)]
    else:
        return False, "Please select an arc curve or connected edge segments."

    if len(arc_edges) < 2:
        return False, "An arc requires at least 2 connected segments."

    ordered_verts = order_chain_vertices(arc_edges)
    if len(ordered_verts) < 3:
        return False, "Could not determine arc vertices sequence."

    v_start = ordered_verts[0]
    v_mid = ordered_verts[len(ordered_verts) // 2]
    v_end = ordered_verts[-1]

    p_start = to_v3d(v_start.position)
    p_mid = to_v3d(v_mid.position)
    p_end = to_v3d(v_end.position)

    ok, center, normal, radius = compute_arc_circumcircle_3d(p_start, p_mid, p_end)
    if not ok or center is None or normal is None or radius < 1e-4:
        return False, "Points are collinear or radius is invalid."

    u_axis = (p_start - center).normalized()
    v_axis = QVector3D.crossProduct(normal, u_axis).normalized()

    v_mid_rel = p_mid - center
    if QVector3D.dotProduct(v_mid_rel, v_axis) < 0:
        normal = -normal
        v_axis = -v_axis

    v_end_rel = p_end - center
    x_end = QVector3D.dotProduct(v_end_rel, u_axis)
    y_end = QVector3D.dotProduct(v_end_rel, v_axis)
    theta_end = math.atan2(y_end, x_end)
    if theta_end <= 0:
        theta_end += 2.0 * math.pi

    if abs(theta_end - 2.0 * math.pi) < 1e-3 or distance_3d(p_start, p_end) < TOLERANCE:
        return False, "The selected curve is already a full closed circle."

    missing_angle = (2.0 * math.pi) - theta_end
    missing_segs = max(int(round(36 * (missing_angle / (2.0 * math.pi)))), 6)

    prev_pos = p_end
    for k in range(1, missing_segs):
        ang = theta_end + (missing_angle * k / missing_segs)
        pt = center + (u_axis * (math.cos(ang) * radius)) + (v_axis * (math.sin(ang) * radius))
        new_e = mesh.add_edge(prev_pos, pt)
        if curve_id is not None:
            new_e.curve = curve_id
        prev_pos = pt

    last_e = mesh.add_edge(prev_pos, p_start)
    if curve_id is not None:
        last_e.curve = curve_id

    if hasattr(mesh, "weld_coincident"):
        try:
            mesh.weld_coincident()
        except Exception:
            pass

    mark_mesh_modified(mesh)
    return True, f"Closed arc into a 360° circle ({missing_segs} segments added, R={radius:.3f}m)."


# =============================================================================
# 4. UNDO / REDO TRANSACTION WRAPPER
# =============================================================================

def execute_transaction(viewport, mutate_func, success_msg: str = "") -> None:
    """Execute a mesh mutation inside IngeTrazo's transactional undo system."""
    try:
        from core.history import SnapshotImport
        viewport.history.execute(SnapshotImport(mutate_func))
        viewport.notify_scene_changed()
        viewport.update()
        if success_msg:
            viewport.flash_status(success_msg, 4000)
    except Exception as exc:
        viewport.flash_status(f"Error: {exc}", 5000)


# =============================================================================
# 5. INTERACTIVE VIEWPORT TOOL (INTERACTIVE PICKING & REAL-TIME PREVIEW)
# =============================================================================

class LinesConnectTool(Tool):
    """Interactive mouse tool for Lines Connect."""
    uses_snap = True
    box_select = False

    def __init__(self, mode: str = "connect"):
        super().__init__()
        self.mode = mode
        self.name = f"Lines Connect ({mode.capitalize()})"
        self.edge1 = None
        self.click_pos1 = None
        self.p1_anchor = None
        self.hover_edge = None
        self.hover_pos = None
        self.preview_lines: List[Tuple[QVector3D, QVector3D]] = []

    def on_activate(self, viewport) -> None:
        global _ACTIVE_TOOL
        _ACTIVE_TOOL = self
        if viewport.active_tool is not self:
            viewport.set_active_tool(self)
            return

        self.reset()
        try:
            viewport.scene.clear_selection()
        except Exception:
            pass
        action_name = "connect" if self.mode == "connect" else "extend to corner"
        viewport.flash_status(
            f"Lines Connect [{self.mode.capitalize()}]: Click Edge 1. (Esc = exit)",
            6000
        )
        viewport.update()

    def on_deactivate(self, viewport) -> None:
        global _ACTIVE_TOOL
        _ACTIVE_TOOL = None
        self.reset()
        try:
            viewport.scene.clear_selection()
        except Exception:
            pass
        viewport.set_hover(None)
        viewport.update()

    def on_cancel(self, viewport) -> None:
        self.reset()
        try:
            viewport.scene.clear_selection()
        except Exception:
            pass
        viewport.flash_status("Lines Connect: cancelled.", 2000)
        viewport.set_active_tool(None)
        viewport.update()

    def on_key(self, viewport, key: int, modifiers: Any) -> bool:
        if key == Qt.Key_Escape:
            self.on_cancel(viewport)
            return True
        return False

    def reset(self) -> None:
        self.edge1 = None
        self.click_pos1 = None
        self.p1_anchor = None
        self.hover_edge = None
        self.hover_pos = None
        self.preview_lines = []

    def rubber_band_lines(self) -> List[Tuple[QVector3D, QVector3D]]:
        return self.preview_lines

    def on_hover(self, ctx) -> None:
        vp = ctx.viewport
        edge = pick_edge_at(vp, ctx.screen.x(), ctx.screen.y())
        self.hover_edge = edge
        self.hover_pos = to_v3d(ctx.world)
        vp.set_hover(edge)

        self._calculate_preview()
        vp.update()

    def on_click(self, ctx) -> None:
        vp = ctx.viewport
        edge = pick_edge_at(vp, ctx.screen.x(), ctx.screen.y())
        if edge is None:
            return

        click_world = to_v3d(ctx.world)

        if self.edge1 is None:
            # 1. Pick First Edge
            self.edge1 = edge
            self.click_pos1 = click_world
            near_v = get_closest_endpoint(edge, click_world)
            self.p1_anchor = to_v3d(near_v.position)

            # Visually highlight Edge 1 in the viewport natively
            try:
                vp.scene.clear_selection()
                vp.scene.select([edge], mode="replace")
            except Exception:
                try:
                    vp.scene.selection.clear()
                    vp.scene.selection.add(edge)
                except Exception:
                    pass

            action_name = "connect" if self.mode == "connect" else "extend"
            vp.flash_status(
                f"Edge 1 selected! Now click Edge 2 (the line in front) to {action_name}.",
                6000
            )
            vp.update()
        else:
            # 2. Pick Second Edge -> Execute!
            if edge == self.edge1:
                return

            edge2 = edge
            click_pos2 = click_world
            status_result = {}

            def mutate(sc):
                if self.mode == "connect":
                    ok, msg = op_connect_edges(
                        sc.mesh, self.edge1, edge2,
                        self.click_pos1, click_pos2
                    )
                else:
                    ok, msg = op_extend_edges(
                        sc.mesh, self.edge1, edge2,
                        self.click_pos1, click_pos2
                    )
                status_result["ok"] = ok
                status_result["msg"] = msg

            execute_transaction(vp, mutate)

            # Clear temporary selection
            try:
                vp.scene.clear_selection()
            except Exception:
                try:
                    vp.scene.selection.clear()
                except Exception:
                    pass

            msg = status_result.get("msg", "Operation completed.")
            vp.flash_status(f"{msg} (Ready for next line, or press Esc to exit)", 5000)

            self.reset()
            vp.update()

    def _calculate_preview(self) -> None:
        self.preview_lines = []
        if self.edge1 is None or self.p1_anchor is None:
            return
        if self.hover_edge is None or self.hover_edge == self.edge1:
            return

        p_A0 = to_v3d(self.edge1.v0.position)
        p_A1 = to_v3d(self.edge1.v1.position)
        p_B0 = to_v3d(self.hover_edge.v0.position)
        p_B1 = to_v3d(self.hover_edge.v1.position)

        v1_full = p_A1 - p_A0
        v2_full = p_B1 - p_B0
        L1 = v1_full.length()
        L2 = v2_full.length()
        if L1 < 1e-6 or L2 < 1e-6:
            return

        v1 = v1_full / L1
        v2 = v2_full / L2

        ok, int_pt, _, (t, u) = intersect_lines_3d_detailed(p_A0, v1, p_B0, v2)

        if ok and int_pt is not None:
            eps = 1e-3
            inside1 = (eps <= t <= L1 - eps)
            inside2 = (eps <= u <= L2 - eps)

            if inside2 and not inside1:
                # T-Junction: Edge 1 extends to Edge 2 and stops right at int_pt!
                tip1 = p_A1 if t > L1 else p_A0
                self.preview_lines = [(tip1, int_pt)]
                return

            elif inside1 and not inside2:
                # T-Junction: Edge 2 extends to Edge 1 and stops right at int_pt!
                tip2 = p_B1 if u > L2 else p_B0
                self.preview_lines = [(tip2, int_pt)]
                return

            elif inside1 and inside2:
                # X-Crossing: both cross inside
                self.preview_lines = [(int_pt, int_pt)]
                return

            elif self.mode == "extend":
                # L-Corner: both lines extend to meet at int_pt!
                tip1 = p_A1 if t > L1 else p_A0
                tip2 = p_B1 if u > L2 else p_B0
                self.preview_lines = [
                    (tip1, int_pt),
                    (tip2, int_pt),
                ]
                return

        # Fallback preview: bridge closest endpoints
        hover_pt = self.hover_pos if self.hover_pos is not None else p_B0
        p2_near = to_v3d(get_closest_endpoint(self.hover_edge, hover_pt).position)
        self.preview_lines = [(self.p1_anchor, p2_near)]


# =============================================================================
# 6. VIEWPORT 2D OVERLAY (BRIGHT HIGHLIGHTS & GLOWING ENDPOINTS)
# =============================================================================

def draw_viewport_overlay(viewport, painter: QPainter) -> None:
    """Draw high-visibility custom overlay points and preview lines in the viewport."""
    tool = _ACTIVE_TOOL
    if tool is None or getattr(viewport, "active_tool", None) is not tool:
        return

    def project_pt(vec: QVector3D) -> Optional[QPointF]:
        try:
            xs, ys, front = viewport.world_to_pixels([[vec.x(), vec.y(), vec.z()]])
            if front[0]:
                return QPointF(float(xs[0]), float(ys[0]))
        except Exception:
            pass
        try:
            return viewport._world_to_pixel(vec)
        except Exception:
            return None

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)

    # 1. Highlight Edge 1 Anchor Endpoint (Orange Circle)
    if tool.p1_anchor is not None:
        pt1_screen = project_pt(tool.p1_anchor)
        if pt1_screen is not None:
            painter.setPen(QPen(QColor(255, 140, 0), 2))
            painter.setBrush(QBrush(QColor(255, 140, 0, 220)))
            painter.drawEllipse(pt1_screen, 6.0, 6.0)

    # 2. Draw Preview Lines (Dashed Bright Orange/Red)
    for start_3d, end_3d in tool.preview_lines:
        s_screen = project_pt(start_3d)
        e_screen = project_pt(end_3d)
        if s_screen is not None and e_screen is not None:
            pen = QPen(QColor(255, 70, 70), 2.5, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(s_screen, e_screen)

            # Intersection point marker
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 60, 60, 240)))
            painter.drawEllipse(e_screen, 5.0, 5.0)

    painter.restore()


# =============================================================================
# 7. TOOL ACTIVATOR & DISPATCHER (HYBRID INTERACTIVE + PRE-SELECTION)
# =============================================================================

def activate_mode(app, mode: str) -> None:
    """Trigger operation: if 2 edges are already selected, run instantly; else start interactive tool."""
    win = getattr(app, "window", None) or getattr(app, "main_window", None)
    vp = app.viewport
    scene = vp.scene
    edges = selected_edges(scene)

    if mode == "arc_circle":
        if edges:
            result = {}
            def mutate(sc):
                ok, msg = op_arc_to_circle(sc.mesh, edges[0], edges)
                result["msg"] = msg
            execute_transaction(vp, mutate)
            vp.flash_status(result.get("msg", "Circle created."), 4000)
        else:
            vp.flash_status("Please select an arc curve first.", 4000)
        return

    if mode == "close_gaps":
        if len(edges) >= 2:
            result = {}
            def mutate(sc):
                count, msg = op_close_selected_gaps(sc.mesh, edges, max_gap=0.05)
                result["msg"] = msg
            execute_transaction(vp, mutate)
            vp.flash_status(result.get("msg", "Gaps closed."), 4000)
        else:
            vp.flash_status("Please select at least 2 edges to close gaps.", 4000)
        return

    # If user explicitly selected exactly 2 edges beforehand, execute immediately
    if len(edges) == 2:
        result = {}
        def mutate(sc):
            if mode == "connect":
                ok, msg = op_connect_edges(sc.mesh, edges[0], edges[1])
            else:
                ok, msg = op_extend_edges(sc.mesh, edges[0], edges[1])
            result["msg"] = msg

        execute_transaction(vp, mutate)
        try:
            scene.clear_selection()
        except Exception:
            try:
                scene.selection.clear()
            except Exception:
                pass
        vp.flash_status(result.get("msg", "Completed."), 4000)
        return

    # Otherwise: Launch the interactive tool fresh and clean!
    try:
        scene.clear_selection()
    except Exception:
        try:
            scene.selection.clear()
        except Exception:
            pass

    interactive_tool = LinesConnectTool(mode)
    vp.set_active_tool(interactive_tool)

    if win is not None:
        for action in getattr(win, "_tool_actions", {}).values():
            action.setChecked(False)

        label = getattr(win, "_tool_label", None)
        if label is not None:
            label.setText(f"Tool: Lines Connect ({mode.capitalize()})")

    vp.setFocus()


# =============================================================================
# 8. HIGH-DPI VECTOR ICON GENERATOR (DARK & LIGHT THEME ADAPTIVE)
# =============================================================================

def draw_vector_icon(icon_type: str) -> QIcon:
    """Render crisp, antialiased, theme-aware vector icons dynamically."""
    pm = QPixmap(48, 48)
    pm.fill(Qt.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    app = QApplication.instance()
    is_dark = True
    if app is not None:
        window_text = app.palette().windowText().color()
        is_dark = window_text.lightness() > 128

    blue_color = QColor(0, 160, 255) if is_dark else QColor(0, 120, 215)
    orange_color = QColor(255, 140, 0)
    red_color = QColor(235, 60, 60)

    pen_solid = QPen(blue_color, 3.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    pen_dashed = QPen(red_color, 2.5, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)

    if icon_type == "connect":
        p.setPen(pen_solid)
        p.drawLine(8, 38, 20, 22)
        p.drawLine(40, 38, 40, 22)

        p.setPen(pen_dashed)
        p.drawLine(20, 22, 40, 22)

        p.setPen(Qt.NoPen)
        p.setBrush(orange_color)
        p.drawEllipse(17, 19, 6, 6)
        p.drawEllipse(37, 19, 6, 6)

    elif icon_type == "extend":
        p.setPen(pen_solid)
        p.drawLine(8, 40, 20, 26)
        p.drawLine(40, 40, 40, 24)

        p.setPen(pen_dashed)
        p.drawLine(20, 26, 40, 6)
        p.drawLine(40, 24, 40, 6)

        p.setPen(Qt.NoPen)
        p.setBrush(orange_color)
        p.drawEllipse(36, 2, 8, 8)

    elif icon_type == "close_gaps":
        p.setPen(pen_solid)
        p.drawLine(6, 24, 18, 24)
        p.drawLine(30, 24, 42, 24)

        p.setPen(pen_dashed)
        p.drawLine(18, 24, 30, 24)

        p.setPen(Qt.NoPen)
        p.setBrush(orange_color)
        p.drawEllipse(15, 21, 6, 6)
        p.drawEllipse(27, 21, 6, 6)

    elif icon_type == "arc_circle":
        p.setPen(pen_solid)
        p.drawArc(8, 8, 32, 32, 0 * 16, 120 * 16)

        p.setPen(pen_dashed)
        p.drawArc(8, 8, 32, 32, 120 * 16, 240 * 16)

        p.setPen(Qt.NoPen)
        p.setBrush(orange_color)
        p.drawEllipse(21, 21, 6, 6)

    p.end()
    return QIcon(pm)


# =============================================================================
# 9. TOOLBAR BUILDER & UI LAYOUT
# =============================================================================

def build_toolbar(app, win, actions_data):
    """Build and register a floating/dockable top toolbar for Lines Connect."""
    toolbar_name = f"toolbar_{KEY}"
    tb = win.findChild(QToolBar, toolbar_name)
    if tb is None:
        tb = QToolBar("Lines Connect", win)
        tb.setObjectName(toolbar_name)
        tb.setMovable(True)
        tb.setFloatable(True)
        tb.setIconSize(QSize(24, 24))
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        win.addToolBar(Qt.TopToolBarArea, tb)

    for icon_key, title, tooltip, callback in actions_data:
        action = QAction(draw_vector_icon(icon_key), title, tb)
        action.setToolTip(f"<b>{title}</b><br>{tooltip}<br><i>Author: {AUTHOR}</i>")
        action.setStatusTip(f"{title} — {tooltip}")
        action.triggered.connect(lambda _checked=False, fn=callback: fn())
        tb.addAction(action)

    return tb


# =============================================================================
# 10. EXTENSION ENTRY POINT (SETUP)
# =============================================================================

def setup(app) -> None:
    """Official IngeTrazo extension entry point."""
    global _GLOBAL_APP
    _GLOBAL_APP = app

    actions_spec = [
        (
            "connect",
            "Connect Edges",
            "Interactive pick or pre-selection: Bridge closest endpoints or connect to transverse line.",
            lambda: activate_mode(app, "connect"),
        ),
        (
            "extend",
            "Extend to Corner / Line",
            "Interactive pick or pre-selection: Extend line to meet line in front or meet at corner.",
            lambda: activate_mode(app, "extend"),
        ),
        (
            "close_gaps",
            "Close Gaps",
            "Batch close open endpoints and small gaps across all selected edges.",
            lambda: activate_mode(app, "close_gaps"),
        ),
        (
            "arc_circle",
            "Arc to Circle",
            "Complete a selected arc along its curvature into a full 360-degree circle.",
            lambda: activate_mode(app, "arc_circle"),
        ),
    ]

    # 1. Extensions Menu Registration
    sub_menu = app.add_menu("Lines Connect")
    if sub_menu is not None:
        for icon_key, title, tooltip, callback in actions_spec:
            act = sub_menu.addAction(draw_vector_icon(icon_key), title)
            act.setToolTip(tooltip)
            act.setStatusTip(f"{title}: {tooltip}")
            act.triggered.connect(lambda _c=False, fn=callback: fn())

        sub_menu.addSeparator()
        info_act = sub_menu.addAction(f"Author: {AUTHOR}")
        info_act.setEnabled(False)
    else:
        for icon_key, title, tooltip, callback in actions_spec:
            app.add_menu_action(f"Lines Connect: {title}", callback, tip=tooltip)

    # 2. Viewport Right-Click Context Menu Hook
    def context_menu_hook(menu: QMenu, selection) -> None:
        edges = selected_edges(app.viewport.scene)
        if not edges:
            return

        menu.addSeparator()
        section_title = menu.addAction(f"— Lines Connect ({AUTHOR}) —")
        section_title.setEnabled(False)

        if len(edges) == 2:
            act_c = menu.addAction(draw_vector_icon("connect"), "Connect Edges")
            act_c.triggered.connect(lambda _c=False: QTimer.singleShot(0, lambda: activate_mode(app, "connect")))

            act_e = menu.addAction(draw_vector_icon("extend"), "Extend to Corner / Line")
            act_e.triggered.connect(lambda _c=False: QTimer.singleShot(0, lambda: activate_mode(app, "extend")))

        if len(edges) >= 2:
            act_g = menu.addAction(draw_vector_icon("close_gaps"), "Close Selected Gaps")
            act_g.triggered.connect(lambda _c=False: QTimer.singleShot(0, lambda: activate_mode(app, "close_gaps")))

        if any(getattr(e, "curve", None) is not None for e in edges):
            act_a = menu.addAction(draw_vector_icon("arc_circle"), "Complete Arc to Circle")
            act_a.triggered.connect(lambda _c=False: QTimer.singleShot(0, lambda: activate_mode(app, "arc_circle")))

    app.add_context_menu(context_menu_hook)

    # 3. Viewport 2D Overlay Drawing Hook (Custom Preview & Endpoints)
    app.add_overlay(draw_viewport_overlay)

    # 4. Top Toolbar Creation
    win = getattr(app, "window", None) or getattr(app, "main_window", None)
    if win is not None:
        build_toolbar(app, win, actions_spec)
