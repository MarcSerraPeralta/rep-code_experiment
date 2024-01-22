from typing import Tuple

from copy import deepcopy

import stim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dem_estimation.utils import stim_to_nx


def plot_dem(
    ax: plt.Axes,
    dem: stim.DetectorErrorModel,
    boundary_shift: Tuple[float, float] = [0.5, 0.5],
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
    probs = list(nx.get_edge_attributes(graph, "prob").values())
    edge_cmap = colormaps["rainbow"]
    edge_vmin, edge_vmax = np.log10(np.min(probs[probs != 0])), np.log10(np.max(probs))

    # draw non-boundary positive edges
    pos = {n: graph.nodes[n]["coords"] for n in real_nodes}
    edge_list = [
        (u, v)
        for u, v in graph.edges()
        if ("boundary" not in f"{u}{v}") and (graph.edges[(u, v)]["prob"] != 0)
    ]
    edge_color = [np.log10(graph.edges[e]["prob"]) for e in edge_list]
    lines = nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edgelist=edge_list,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
    )

    # draw positive boundary edges
    pos = {n: graph.nodes[n]["coords"] for n in graph.nodes()}
    edge_list = [
        (u, v)
        for u, v in graph.edges()
        if ("boundary" in f"{u}{v}") and (graph.edges[(u, v)]["prob"] != 0)
    ]
    edge_color = [np.log10(graph.edges[e]["prob"]) for e in edge_list]
    lines = nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edgelist=edge_list,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
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

    # draw non-boundary edges
    pos = {n: graph1.nodes[n]["coords"] for n in real_nodes}
    edge_list = [(u, v) for u, v in graph1.edges() if "boundary" not in f"{u}{v}"]
    edge_color = [diff[k] for k, e in enumerate(edges) if e in edge_list]
    lines = nx.draw_networkx_edges(
        graph1,
        pos=pos,
        ax=ax,
        edgelist=edge_list,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
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
    )

    # Add a colorbar to the right of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4)
    colorbar = plt.colorbar(lines, cax=cax)
    colorbar.set_label("$p_1 - p_2$")
    colorbar.ax.yaxis.set_label_position("left")

    return ax
