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

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/defect_analysis"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

DEFECTS_NAME = "defects_DecayLinearClassifierFit"

####################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")

    print(f"\033[F\033[K{config_data['data'].format(**element)}", flush=True)

    # load defect data
    defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
    defects = defects_xr.defects
    final_defects = defects_xr.final_defects

    # sort defect data into vector with same ordering
    # as the stim circuit
    defect_vec = get_defect_vector(
        defects,
        final_defects,
        anc_order=layout.get_qubits(role="anc"),
        dim_first="qec_round",
    )

    # obtain and plot the Pij matrix
    pij = get_pij_matrix(defect_vec)

    fig, ax = plt.subplots()

    plot_pij_matrix(
        ax=ax,
        pij=pij,
        qubit_labels=layout.get_qubits(role="anc"),
        num_rounds=element["num_rounds"] + 1,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"pij_{DEFECTS_NAME}.pdf", format="pdf")
    plt.close()
