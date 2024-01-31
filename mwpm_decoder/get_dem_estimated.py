import pathlib
import os
import yaml
from copy import deepcopy

import numpy as np
import xarray as xr
import stim

from dem_estimation import get_edge_probabilities
from dem_estimation.utils import (
    stim_to_edges,
    edges_to_stim,
    floor_boundary_edges,
    clip_negative_edges,
)
from qec_util import Layout
from rep_code.defects import get_defect_vector
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/dems"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

DEFECTS_NAME = "defects_TwoStateLinearClassifierFit"
EDGES_FROM_NOISE = "t1t2_noise"

####################

NOISE_NAME = "estimated_noise"

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

list_states = config_data["string_data_options"].pop("state")

new_config_data = deepcopy(config_data)
new_config_data["data"] = new_config_data["data"].replace("_s{state}", "")
new_config_data["config"] = new_config_data["config"].replace("_s{state}", "")
STRING_DATA = new_config_data["string_data_options"]

(OUTPUT_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / EXP_NAME / "config_data.yaml", "w") as file:
    yaml.dump(new_config_data, file, default_flow_style=False)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = OUTPUT_DIR / EXP_NAME / new_config_data["config"].format(**element)
    dem_dir = OUTPUT_DIR / EXP_NAME / new_config_data["data"].format(**element)
    print(f"\033[F\033[K{dem_dir}", flush=True)

    # load model and layout (should be present from t1t2 noise)
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")

    # load all defect data
    list_datasets = []
    for state in list_states:
        data_dir = (
            DATA_DIR / EXP_NAME / config_data["data"].format(**element, state=state)
        )
        defects = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
        list_datasets.append(defects)

    # combine datasets
    full_data = xr.concat(list_datasets, dim="shot")
    full_data["shot"] = list(range(len(full_data.shot)))

    defects = full_data.defects
    final_defects = full_data.final_defects
    log_flips = full_data.log_flips.values

    # sort defect data into vector with same ordering
    # as the stim circuit
    defect_vec = get_defect_vector(
        defects,
        final_defects,
        anc_order=layout.get_qubits(role="anc"),
        dim_first="anc_qubit",
    )

    # get t1/t2+classical meas circuit
    dem_with_edges = stim.DetectorErrorModel.from_file(
        dem_dir / f"{EDGES_FROM_NOISE}.dem"
    )
    edges, boundary_edges, edge_logicals, detector_coords = stim_to_edges(
        dem_with_edges, return_coords=True
    )

    # get decoding graph including
    # edges and boundary edges
    dem = get_edge_probabilities(
        defect_vec,
        edges=list(edges.keys()),
        boundary_edges=list(boundary_edges.keys()),
    )

    # floor boundary edges and ensure positive probabilities
    dem = floor_boundary_edges(dem, boundary_edges)
    dem = clip_negative_edges(dem)

    # to stim DEM format
    dem = edges_to_stim(dem, edge_logicals, detector_coords=detector_coords)
    dem.to_file(dem_dir / f"{NOISE_NAME}_{DEFECTS_NAME.replace('defects_', '')}.dem")
