print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from dem_estimation.utils import get_pij_matrix
from dem_estimation.plots import plot_pij_matrix
from qec_util import Layout
from rep_code.defects import get_defect_vector
from rep_code.dataset import sequence_generator
from rep_code.circuits.repetition_code import get_1d_coords

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/defect_analysis"
)

EXP_NAME = "20230119_initial_data_d3"

DEFECTS_NAME = "defects_DecayLinearClassifierFit"

####################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
STATES = STRING_DATA.pop("state")

# remove state from directory names
new_config_data = deepcopy(config_data)
new_config_data["string_data_options"] = STRING_DATA
new_config_data["data"] = new_config_data["data"].replace("_s{state}", "")
new_config_data["config"] = new_config_data["config"].replace("_s{state}", "")
new_config_data["readout_calibration"] = new_config_data["readout_calibration"].replace(
    "_s{state}", ""
)

(OUTPUT_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / EXP_NAME / "config_data.yaml", "w") as file:
    yaml.dump(new_config_data, file, default_flow_style=False)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")

    defect_vec_combined = []

    for state in STATES:
        data_dir = (
            DATA_DIR / EXP_NAME / config_data["data"].format(**element, state=state)
        )
        print(f"\033[F\033[K{data_dir}", flush=True)

        # load defect data
        defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
        defects = defects_xr.defects
        final_defects = defects_xr.final_defects

        # sort defect data into vector with ordering
        # that matches the chain that they define
        coords_dict = get_1d_coords(layout)
        anc_order = sorted(
            layout.get_qubits(role="anc"),
            key=lambda q: coords_dict[q],
        )
        defect_vec = get_defect_vector(
            defects,
            final_defects,
            anc_order=anc_order,
            dim_first="qec_round",
        )

        # obtain and plot the Pij matrix
        pij = get_pij_matrix(defect_vec)

        fig, ax = plt.subplots()

        plot_pij_matrix(
            ax=ax,
            pij=pij,
            qubit_labels=anc_order,
            num_rounds=element["num_rounds"] + 1,
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"{DEFECTS_NAME}_pij_s{state}.pdf", format="pdf")
        plt.close()

        defect_vec_combined.append(defect_vec)

    # combine data
    defect_vec_combined = np.concatenate(defect_vec_combined, axis=0)

    # sort defect data into vector with ordering
    # that matches the chain that they define
    coords_dict = get_1d_coords(layout)
    anc_order = sorted(
        layout.get_qubits(role="anc"),
        key=lambda q: coords_dict[q],
    )

    # obtain and plot the Pij matrix
    pij = get_pij_matrix(defect_vec_combined)

    fig, ax = plt.subplots()

    plot_pij_matrix(
        ax=ax,
        pij=pij,
        qubit_labels=anc_order,
        num_rounds=element["num_rounds"] + 1,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"{DEFECTS_NAME}_pij_combined.pdf", format="pdf")
    plt.close()
