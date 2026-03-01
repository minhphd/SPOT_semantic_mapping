"""
Scene Graph Visualization with Interactive Room Annotation
==========================================================

This module provides an interactive 2D top-down visualization of a 3D scene
graph overlaid on a manually aligned floorplan. It enables:

• Visualization of object nodes (x–z projection)
• Visualization of relational edges with hover metadata
• Manual floorplan alignment (offset + rotation)
• Interactive room polygon drawing in the browser
• Automatic room naming via prompt
• Export of room annotations to rooms.json
• Post-processing assignment of objects to rooms in Python

Designed for embodied scene graph pipelines where objects are reconstructed
in metric 3D space (e.g., RGB-D fusion or SLAM outputs) and require semantic
room segmentation.

---------------------------------------------------------------------

Expected Tracker Interface
--------------------------
The `tracker` object must expose:

    tracker.objects  → iterable of objects with:
        - oid
        - class_name
        - room (string, optional)
        - bbox.get_center() → (x, y, z)

    tracker.edges    → iterable of edges with:
        - src_id
        - dst_id
        - rtype
        - score
        - dist
        - src
        - dst

Only (x, z) coordinates are used for top-down visualization.

---------------------------------------------------------------------

Main Function
-------------
plot_scene_graph_over_floorplan_manual(...)

Generates an interactive HTML file that allows:

• Drawing room polygons (closed paths)
• Prompting for room names
• Exporting annotated rooms as rooms.json

Floorplan Alignment Parameters
------------------------------
FLOOR_OFFSET_X, FLOOR_OFFSET_Y
    Manual translation in meters.

FLOOR_ROTATION_DEG
    Counter-clockwise rotation of floorplan in degrees.

resolution
    Meters per pixel of the floorplan image.

---------------------------------------------------------------------

Room Assignment Utilities
-------------------------
assign_rooms_from_json(tracker, rooms_json)

Loads exported rooms.json and assigns each object to the first polygon
containing its (x, z) center using a ray-casting point-in-polygon test.

---------------------------------------------------------------------

Coordinate Conventions
----------------------
• World frame assumed metric (meters)
• Top-down uses (x, z)
• Y-axis inverted in Plotly for image alignment
• Floorplan image vertically flipped for proper orientation

---------------------------------------------------------------------

Output
------
• Interactive HTML viewer (default: scene_graph_2d.html)
• rooms.json (via browser button)
• Updated tracker objects with room labels (via Python function)

---------------------------------------------------------------------

Typical Workflow
----------------
1. Generate scene graph from reconstruction pipeline
2. Call plot_scene_graph_over_floorplan_manual(...)
3. Draw rooms in browser and export rooms.json
4. Run assign_rooms_from_json(...)
5. Re-run visualization with room labels

---------------------------------------------------------------------

Dependencies
------------
• NumPy
• NetworkX
• Plotly
• Pillow
• JSON
• re

---------------------------------------------------------------------

Use Case
--------
Ideal for embodied AI research, robotic mapping, indoor localization,
semantic scene graph construction, and room-level reasoning tasks.
"""


import json
import re
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from PIL import Image


# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================

def plot_scene_graph_over_floorplan_manual(
    tracker,
    floorplan_path="floorplan.png",
    resolution=0.05,
    outfile="scene_graph_2d.html",
    node_size=12,

    # --------------------------------
    # MANUAL FLOORPLAN ALIGNMENT PARAMS
    # --------------------------------
    FLOOR_OFFSET_X=-57,
    FLOOR_OFFSET_Y=-12,
    FLOOR_ROTATION_DEG=0,
):
    """
    Interactive scene graph viewer:
    - Nodes plotted in (x,z) top-down
    - Floorplan manually aligned
    - User draws room polygons
    - Prompt asks for room name
    - rooms.json export button included
    """

    # -------------------------------------------------------
    # Load floorplan
    # -------------------------------------------------------
    floor = Image.open(floorplan_path)
    floor = floor.transpose(method=Image.FLIP_TOP_BOTTOM)
    floor = Image.eval(floor, lambda x: 255 - x)
    W, H = floor.size

    floor_w_m = W * resolution
    floor_h_m = H * resolution

    # -------------------------------------------------------
    # Build scene graph
    # -------------------------------------------------------
    G = nx.DiGraph()

    for obj in tracker.objects:
        cx, cy, cz = obj.bbox.get_center()
        G.add_node(
            obj.oid,
            x=cx,
            y=cz,
            cls=obj.class_name,
            room=obj.room
        )

    for e in tracker.edges:
        G.add_edge(
            e.src_id,
            e.dst_id,
            rtype=e.rtype,
            score=e.score,
            dist=e.dist,
            src=e.src,
            dst=e.dst,
        )

    print(G.edges(data=True))
    # -------------------------------------------------------
    # Plotly figure
    # -------------------------------------------------------
    fig = go.Figure()

    # -------------------------------------------------------
    # Floorplan transform
    # -------------------------------------------------------
    angle = np.deg2rad(FLOOR_ROTATION_DEG)
    corners = np.array([
        [0, 0],
        [floor_w_m, 0],
        [floor_w_m, floor_h_m],
        [0, floor_h_m],
    ])

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    rotated = corners @ R.T
    rotated[:, 0] += FLOOR_OFFSET_X
    rotated[:, 1] += FLOOR_OFFSET_Y

    fig.add_layout_image(
        dict(
            source=floor,
            xref="x",
            yref="y",
            x=rotated[0, 0],
            y=rotated[0, 1],
            sizex=floor_w_m,
            sizey=floor_h_m,
            sizing="stretch",
            opacity=0.75,
            layer="below",
        )
    )

    # -------------------------------------------------------
    # Edges
    # -------------------------------------------------------
    for u, v, d in G.edges(data=True):
        fig.add_trace(go.Scatter(
            x=[G.nodes[u]["x"], G.nodes[v]["x"]],
            y=[G.nodes[u]["y"], G.nodes[v]["y"]],
            mode="lines",
            line=dict(color="cyan", width=2),
            hoverinfo="text",
            hovertext=(
                f"<b>{d['rtype']}</b><br>"
                f"score: {d['score']:.3f}<br>"
                f"dist: {d['dist']:.2f}<br>"
                f"{d['src']} → {d['dst']}"
            )
        ))

    # -------------------------------------------------------
    # Nodes
    # -------------------------------------------------------
    X, Y, H, C = [], [], [], []

    classes = [G.nodes[n]["cls"] for n in G.nodes]
    uniq = sorted(set(classes))
    cmap = {c: f"hsl({i*(360//len(uniq))},80%,50%)" for i, c in enumerate(uniq)}

    for n, d in G.nodes(data=True):
        X.append(d["x"])
        Y.append(d["y"])
        C.append(cmap[d["cls"]])
        H.append(
            f"<b>{n}</b><br>"
            f"class: {d['cls']}<br>"
            f"pos: ({d['x']:.2f}, {d['y']:.2f})<br>"
            f"room: {d['room']}"
        )

    fig.add_trace(go.Scatter(
        x=X,
        y=Y,
        mode="markers",
        marker=dict(size=node_size, color=C),
        hoverinfo="text",
        hovertext=H,
    ))

    # -------------------------------------------------------
    # Layout + drawing tools
    # -------------------------------------------------------
    fig.update_layout(
        width=1800,
        height=1400,
        title="Scene Graph with Manual Room Annotation",
        showlegend=False,
        dragmode="drawclosedpath",
        newshape=dict(
            line=dict(color="magenta", width=3),
            fillcolor="rgba(255,0,255,0.15)",
        ),
        xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
    )

    fig.write_html(outfile, include_plotlyjs="cdn")
    append_room_prompt_js(outfile)
    print(f"[saved] {outfile}")


# ============================================================
# JAVASCRIPT: prompt for room name + export
# ============================================================

def append_room_prompt_js(outfile):
    js = r"""
<script>
(function() {
  const wait = setInterval(() => {
    const gd = document.querySelector('.js-plotly-plot');
    if (!gd) return;
    clearInterval(wait);

    gd.on('plotly_relayout', () => {
      const shapes = gd.layout.shapes || [];
      const last = shapes[shapes.length - 1];
      if (!last || !last.path || last.meta?.room) return;

      const room = prompt("Room name (e.g. kitchen, bedroom, hallway):");
      if (!room) return;

      last.meta = last.meta || {};
      last.meta.room = room;
      Plotly.relayout(gd, { shapes });
    });

    const btn = document.createElement('button');
    btn.innerText = 'Download rooms.json';
    btn.style.position = 'fixed';
    btn.style.top = '12px';
    btn.style.right = '12px';
    btn.style.zIndex = 9999;
    btn.style.padding = '10px 12px';

    btn.onclick = function() {
      const shapes = gd.layout.shapes || [];
      const rooms = shapes.map(s => ({
        room: s.meta?.room || "unnamed",
        path: s.path
      }));

      const blob = new Blob(
        [JSON.stringify({ rooms }, null, 2)],
        { type: "application/json" }
      );
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "rooms.json";
      a.click();
      URL.revokeObjectURL(a.href);
    };

    document.body.appendChild(btn);
  }, 200);
})();
</script>
"""
    with open(outfile, "a", encoding="utf-8") as f:
        f.write(js)

# ============================================================
# PYTHON: assign rooms to objects
# ============================================================
def parse_room_path(path):
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", path)))
    return np.array(list(zip(nums[0::2], nums[1::2])))

def inside(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
            (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def assign_rooms_from_json(tracker, rooms_json):

    with open(rooms_json) as f:
        rooms = json.load(f)["rooms"]

    parsed = [(r["room"], parse_room_path(r["path"])) for r in rooms]

    for obj in tracker.objects:
        cx, _, cz = obj.bbox.get_center()
        obj.room = "unassigned"
        for name, poly in parsed:
            if inside(cx, cz, poly):
                obj.room = name
                break
