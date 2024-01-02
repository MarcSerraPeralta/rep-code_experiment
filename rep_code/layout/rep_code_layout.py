from typing import List
import importlib_resources as impresources

from copy import deepcopy
import pathlib

import yaml
import networkx as nx

from qec_util import Layout

from .. import layout as layout_repo


def get_rep_code_layout(qubits: List[str]):
    filename = impresources.files(layout_repo) / "repetition_code_layout.yaml"
    with open(filename, "r") as file:
        layout_setup = yaml.safe_load(file)

    new_layout = deepcopy(layout_setup)
    new_layout["layout"] = []

    for qubit_dict in layout_setup["layout"]:
        q_label = qubit_dict["qubit"]
        if q_label not in qubits:
            continue

        # remove non-used qubits from the neighbours
        qubit_dict["neighbors"] = {
            o: q if q in qubits else None for o, q in qubit_dict["neighbors"].items()
        }

        # add to layout
        new_layout["layout"].append(qubit_dict)

    # update distance
    data_qubits = [q["qubit"] for q in new_layout["layout"] if q["role"] == "data"]
    distance = len(data_qubits)
    new_layout["distance"] = distance

    layout = Layout(new_layout)

    for qubit in data_qubits[1:]:
        if not nx.has_path(layout.graph, data_qubits[0], qubit):
            raise ValueError(
                f"Selection of qubits does not lead to a repetition code: {qubits}"
            )

    return layout
