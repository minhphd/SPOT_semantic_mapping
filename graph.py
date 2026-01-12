import plotly.graph_objects as go
import networkx as nx
import numpy as np
from PIL import Image


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
        FLOOR_ROTATION_DEG=0
):
    """
    Plot scene graph using (x, z) top-down, with floorplan manually aligned.
    """

    # -------------------------------------------------------
    # Load floorplan
    # -------------------------------------------------------
    floor = Image.open(floorplan_path)
    # mirror the floorplan
    floor = floor.transpose(method=Image.FLIP_TOP_BOTTOM)
    # invert floorplan colors
    floor = Image.eval(floor, lambda x: 255 - x)
    W, H = floor.size

    # Convert floorplan to meters
    floor_w_m = W * resolution
    floor_h_m = H * resolution

    # -------------------------------------------------------
    # Extract world coords (top-down projection)
    # -------------------------------------------------------
    # xs = []
    # ys = []  # from Z

    # for obj in tracker.objects:
    #     cx, cy, cz = obj.bbox.get_center()
    #     xs.append(cx)
    #     ys.append(cz)

    # xs = np.array(xs)
    # ys = np.array(ys)

    # graph extent (used only for auto-scaling plot)
    # min_x, max_x = xs.min(), xs.max()
    # min_y, max_y = ys.min(), ys.max()

    # -------------------------------------------------------
    # Build Graph (in top-down coords)
    # -------------------------------------------------------
    G = nx.DiGraph()

    for obj in tracker.objects:
        cx, cy, cz = obj.bbox.get_center()
        tx, ty = cx, cz  # top-down projection
        G.add_node(
            obj.oid,
            x=tx,
            y=ty,
            cls=obj.class_name,
        )

    for e in tracker.edges:
        G.add_edge(
            e.src_id,
            e.dist_id,
            src_name = e.src,
            dst_name = e.dst,
            rtype=e.rtype,
            score=e.score,
            dist=e.dist,
        )

    # -------------------------------------------------------
    # Plotting
    # -------------------------------------------------------
    fig = go.Figure()

    # -------------------------------------------------------
    # FLOORPLAN OVERLAY (MANUAL ALIGNMENT)
    # -------------------------------------------------------
    # Place floorplan as a rectangle in *world coordinates*
    # Apply rotation + translation manually

    angle = np.deg2rad(FLOOR_ROTATION_DEG)

    # floorplan corners before rotation
    corners = np.array([
        [0, 0],
        [floor_w_m, 0],
        [floor_w_m, floor_h_m],
        [0, floor_h_m],
    ])

    # apply rotation
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = corners @ R.T

    # apply translation
    rotated[:, 0] += FLOOR_OFFSET_X
    rotated[:, 1] += FLOOR_OFFSET_Y

    xs_floor = rotated[:, 0]
    ys_floor = rotated[:, 1]

    # Plot the image using an affine transform
    fig.add_layout_image(
        dict(
            source=floor,
            xref="x",
            yref="y",
            x=xs_floor[0],
            y=ys_floor[0],
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
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]["x"], G.nodes[u]["y"]
        x1, y1 = G.nodes[v]["x"], G.nodes[v]["y"]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=2, color="cyan"),
            hovertemplate=(
                "<b>Relation:</b> %{customdata[0]}<br>"
                "<b>Score:</b> %{customdata[1]:.3f}<br>"
                "<b>Distance:</b> %{customdata[2]:.3f}<br>"
                "<b>Src:</b> %{customdata[3]}<br>"
                "<b>Dst:</b> %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            customdata=[[data["rtype"], data["score"], data["dist"], data['src_name'], data['dst_name']]]
        ))

    # -------------------------------------------------------
    # Nodes
    # -------------------------------------------------------
    Xp = []
    Yp = []
    hovertext = []
    classes = []

    for nid, nd in G.nodes(data=True):
        Xp.append(nd["x"])
        Yp.append(nd["y"])
        hovertext.append(
            f"<b>{nid}</b><br>"
            f"class: {nd['cls']}<br>"
            f"pos: ({nd['x']:.2f}, {nd['y']:.2f})"
        )
        classes.append(nd["cls"])

    # color by class
    unique = sorted(set(classes))
    class_to_color = {
        c: f"hsl({i*(360//len(unique))},80%,50%)"
        for i, c in enumerate(unique)
    }
    colors = [class_to_color[c] for c in classes]

    fig.add_trace(go.Scatter(
        x=Xp,
        y=Yp,
        mode="markers",
        marker=dict(size=node_size, color=colors),
        hoverinfo="text",
        hovertext=hovertext,
    ))

    # -------------------------------------------------------
    # Layout
    # -------------------------------------------------------
    fig.update_layout(
        width=1800,
        height=1400,
        title="Scene Graph (X,Z) with Manual Floorplan Alignment",
        showlegend=False,
        xaxis=dict(
            scaleanchor="y",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",  # top-down view
            showgrid=False,
            zeroline=False
        ),
    )

    fig.write_html(outfile, include_plotlyjs="cdn")
    print(f"[saved] {outfile}")
