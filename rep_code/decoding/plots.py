from typing import Tuple, List

from copy import deepcopy
import math

import stim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import to_rgba
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dem_estimation.utils import stim_to_nx


def draw_edge(
    p1: np.ndarray, p2: np.ndarray, curve_radius, color, alpha=1.0
) -> FancyArrowPatch:
    # Function to draw curved edges
    direction = p2 - p1
    norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
    orthogonal_direction = np.array([-direction[1], direction[0]]) / norm
    control_point = (p1 + p2) / 2 + orthogonal_direction * curve_radius
    path = Path([p1, control_point, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-",
        color=color,
        alpha=alpha,
        lw=2,
        connectionstyle="arc3,rad=0.2",
    )
    return patch


def draw_edges(
    graph: nx.Graph,
    ax: plt.Axes,
    edgelist: List,
    edge_values: List,
    edge_cmap,
    edge_vmin: float,
    edge_vmax: float,
    add_text: bool = True,
    color=None,
) -> List:
    lines = []
    curve_weight = 0.0
    long_curve_weight = 0.6
    real_nodes = [n for n in graph.nodes() if "boundary" not in str(n)]
    index_to_position_vec = {n: graph.nodes[n]["coords"] for n in real_nodes}

    # Create a color map based on the edge probability
    norm = plt.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=edge_cmap)

    for index, ((u, v), (edge_value)) in enumerate(zip(edgelist, edge_values)):
        if color is None:
            color_edge = mappable.to_rgba(edge_value)
        else:
            color_edge = color

        p1 = np.array(index_to_position_vec[u])
        p2 = np.array(index_to_position_vec[v])
        measurement_edge = np.abs(p1 - p2)[1] == 2
        in_even_round = p1[1] % 2 == 0

        total_weight = curve_weight
        if measurement_edge:
            total_weight = long_curve_weight
        if in_even_round:
            total_weight *= -1

        line: FancyArrowPatch = draw_edge(p1, p2, total_weight, color_edge, alpha=1.0)
        ax.add_patch(line)
        lines.append(line)

        if add_text:
            center = (p1 + p2) / 2
            angle = calculate_angle(p1, p2)
            angle_rad = math.radians(angle)

            # Apply default text position offset
            relative_y_offset = 0.1
            if measurement_edge:
                relative_y_offset = -0.35
            if in_even_round:
                relative_y_offset *= -1

            # Calculate offset
            offset_x: float = relative_y_offset * math.sin(angle_rad)
            offset_y: float = relative_y_offset * math.cos(angle_rad)

            # Apply offset to text position
            text_x = center[0] + offset_x
            text_y = center[1] + offset_y

            ax.text(
                s=f"{edge_value:0.2f}",
                x=text_x,  # center[0],
                y=text_y,  # center[1],
                rotation=angle,
                ha="center",
                va="center",
            )

    return lines


def calculate_angle(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate the angle with the x-axis for the line between (x1, y1) and (x2, y2)."""
    angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

    # Adjust angle to be within the range pi/2 <= angle < -pi/2
    if -90 < angle <= 90:
        return angle
    else:
        # Add or subtract 180 degrees to flip the text
        return (angle + 180) % 360


def plot_dem(
    ax: plt.Axes,
    dem: stim.DetectorErrorModel,
    boundary_shift: Tuple[float, float] = [0.5, 0.5],
    add_text: bool = True,
) -> plt.Axes:
    boundary_shift = np.array(boundary_shift)
    graph = stim_to_nx(dem)

    # create boundary nodes for each boundary edge
    list_edges = deepcopy(graph.edges(data=True))  # avoid modifing dictionary in loop
    for node1, node2, attrs in list_edges:
        if node2 == "boundary":
            new_boundary = f"boundary{node1}"
            graph.add_node(
                new_boundary, coords=graph.nodes[node1]["coords"] + boundary_shift
            )
            graph.add_edge(node1, new_boundary, **attrs)
    graph.remove_node("boundary")

    # draw real nodes
    real_nodes = [n for n in graph.nodes() if "boundary" not in str(n)]
    labels = {n: str(n) for n in real_nodes}
    pos = {n: graph.nodes[n]["coords"] for n in real_nodes}
    nx.draw_networkx_nodes(
        graph, pos=pos, nodelist=real_nodes, node_size=300, ax=ax, node_color="black"
    )
    nx.draw_networkx_labels(
        graph, pos=pos, labels=labels, ax=ax, font_color="white", font_size=10
    )

    # draw boundary node as separated nodes
    b_nodes = [n for n in graph.nodes() if "boundary" in str(n)]
    labels = {n: n.replace("boundary", "b") for n in b_nodes}
    pos = {n: graph.nodes[n]["coords"] for n in b_nodes}
    nx.draw_networkx_nodes(
        graph, pos=pos, nodelist=b_nodes, node_size=200, ax=ax, node_color="gray"
    )
    nx.draw_networkx_labels(
        graph, pos=pos, labels=labels, ax=ax, font_color="white", font_size=6
    )

    # set up colors for edges
    edges = graph.edges()
    probs = np.array([nx.get_edge_attributes(graph, "prob")[e] for e in edges])
    edge_cmap = colormaps["rainbow"]
    edge_vmin = np.min(np.log10(probs[probs != 0]))
    edge_vmax = np.max(np.log10(probs[probs != 0]))

    # Draw edges with curves
    edge_list = [(u, v) for u, v in graph.edges() if "boundary" not in f"{u}{v}"]
    edge_values = [graph.edges[e]["prob"] for e in edge_list]
    edge_list_positive = [e for e, p in zip(edge_list, edge_values) if p > 0]
    edge_values_positive = [
        np.log10(p) for e, p in zip(edge_list, edge_values) if p > 0
    ]
    lines = draw_edges(
        graph=graph,
        ax=ax,
        edgelist=edge_list_positive,
        edge_values=edge_values_positive,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        add_text=add_text,
    )
    edge_list_zero = [e for e, p in zip(edge_list, edge_values) if p == 0]
    edge_values_zero = [None for e, p in zip(edge_list, edge_values) if p == 0]
    draw_edges(
        graph=graph,
        ax=ax,
        edgelist=edge_list_zero,
        edge_values=edge_values_zero,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        add_text=False,
        color="gray",
    )

    # draw boundary edges
    pos = {n: graph.nodes[n]["coords"] for n in graph.nodes()}
    edge_list = [(u, v) for u, v in graph.edges() if "boundary" in f"{u}{v}"]
    edge_color = [graph.edges[e]["prob"] for e in edge_list]
    edge_list_positive = [e for e, p in zip(edge_list, edge_values) if p > 0]
    edge_values_positive = [
        np.log10(p) for e, p in zip(edge_list, edge_values) if p > 0
    ]
    lines = nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edgelist=edge_list_positive,
        edge_color=edge_values_positive,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
    )

    edge_list_zero = [e for e, p in zip(edge_list, edge_values) if p == 0]
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edgelist=edge_list_zero,
        edge_color="gray",
    )

    # Add a colorbar to the right of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4)
    colorbar = plt.colorbar(lines, cax=cax)
    colorbar.set_label("$log_{10}(p_{edge})$")
    colorbar.ax.yaxis.set_label_position("left")

    return ax


def plot_dem_difference(
    ax: plt.Axes,
    dem1: stim.DetectorErrorModel,
    dem2: stim.DetectorErrorModel,
    boundary_shift: Tuple[float, float] = [0.5, 0.5],
    add_text: bool = True,
) -> plt.Axes:
    boundary_shift = np.array(boundary_shift)
    graph1, graph2 = stim_to_nx(dem1), stim_to_nx(dem2)

    assert set(graph1.nodes()) == set(graph2.nodes())
    assert set(graph1.edges()) == set(graph2.edges())

    # create boundary nodes for each boundary edge
    list_edges = deepcopy(graph1.edges(data=True))  # avoid modifing dictionary in loop
    for node1, node2, attrs in list_edges:
        if node2 == "boundary":
            new_boundary = f"boundary{node1}"
            graph1.add_node(
                new_boundary, coords=graph1.nodes[node1]["coords"] + boundary_shift
            )
            graph1.add_edge(node1, new_boundary, **attrs)
    graph1.remove_node("boundary")

    list_edges = deepcopy(graph2.edges(data=True))  # avoid modifing dictionary in loop
    for node1, node2, attrs in list_edges:
        if node2 == "boundary":
            new_boundary = f"boundary{node1}"
            graph2.add_node(
                new_boundary, coords=graph2.nodes[node1]["coords"] + boundary_shift
            )
            graph2.add_edge(node1, new_boundary, **attrs)
    graph2.remove_node("boundary")

    # draw real nodes
    real_nodes = [n for n in graph1.nodes() if "boundary" not in str(n)]
    labels = {n: str(n) for n in real_nodes}
    pos = {n: graph1.nodes[n]["coords"] for n in real_nodes}
    nx.draw_networkx_nodes(
        graph1, pos=pos, nodelist=real_nodes, node_size=300, ax=ax, node_color="black"
    )
    nx.draw_networkx_labels(
        graph1, pos=pos, labels=labels, ax=ax, font_color="white", font_size=10
    )

    # draw boundary node as separated nodes
    b_nodes = [n for n in graph1.nodes() if "boundary" in str(n)]
    labels = {n: n.replace("boundary", "b") for n in b_nodes}
    pos = {n: graph1.nodes[n]["coords"] for n in b_nodes}
    nx.draw_networkx_nodes(
        graph1, pos=pos, nodelist=b_nodes, node_size=200, ax=ax, node_color="gray"
    )
    nx.draw_networkx_labels(
        graph1, pos=pos, labels=labels, ax=ax, font_color="white", font_size=6
    )

    # set up colors for edges
    edges = graph1.edges()
    diff = [
        nx.get_edge_attributes(graph1, "prob")[e]
        - nx.get_edge_attributes(graph2, "prob")[e]
        for e in edges
    ]
    edge_cmap = colormaps["rainbow"]
    edge_vmin, edge_vmax = np.min(diff), np.max(diff)

    # Draw edges with curves
    edge_list = [(u, v) for u, v in graph1.edges() if "boundary" not in f"{u}{v}"]
    edge_values = [diff[k] for k, e in enumerate(edges) if e in edge_list]
    edge_cmap = colormaps["rainbow"]
    lines = draw_edges(
        graph=graph1,
        ax=ax,
        edgelist=edge_list,
        edge_values=edge_values,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        add_text=add_text,
    )

    # draw boundary edges
    pos = {n: graph1.nodes[n]["coords"] for n in graph1.nodes()}
    edge_list = [(u, v) for u, v in graph1.edges() if "boundary" in f"{u}{v}"]
    edge_color = [diff[k] for k, e in enumerate(edges) if e in edge_list]
    lines = nx.draw_networkx_edges(
        graph1,
        pos=pos,
        ax=ax,
        edgelist=edge_list,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
    )

    # Add a colorbar to the right of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4)
    colorbar = plt.colorbar(lines, cax=cax)
    colorbar.set_label("$p_1 - p_2$")
    colorbar.ax.yaxis.set_label_position("left")

    return ax
