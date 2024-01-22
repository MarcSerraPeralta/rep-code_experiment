import pathlib
import os
import yaml

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

EXP_NAME = "20230119_initial_data_d3_s010"

DEFECTS_NAME = "defects_TwoStateLinearClassifierFit"
EDGES_FROM_NOISE = "t1t2_noise"

####################

NOISE_NAME = "estimated_noise"

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)

    # load model and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")

    print(f"\033[F\033[K{config_data['data'].format(**element)}", flush=True)

    # load defect data
    defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
    defects = defects_xr.defects
    final_defects = defects_xr.final_defects
    log_flips = defects_xr.log_flips.values

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
        data_dir / f"{EDGES_FROM_NOISE}.dem"
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
    dem.to_file(data_dir / f"{NOISE_NAME}_{DEFECTS_NAME.replace('defects_', '')}.dem")
