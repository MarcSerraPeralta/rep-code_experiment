import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import stim

from qec_util import Layout
from surface_sim import Setup
from rep_code.dataset import sequence_generator
from rep_code.circuits import memory_experiment
from rep_code.models import *

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

CONFIG_DATA = "config_data.yaml"
NUM_SHOTS = 100_000

LAYOUT_NAME = "rep_code_layout.yaml"
DEM_NAME = "estimated_noise_DecayLinearClassifierFit"

###############################

with open(DATA_DIR / EXP_NAME / CONFIG_DATA, "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
CONFIG_DIR = pathlib.Path("configs")

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"{config_data['data'].format(**element)}", end="\r")

    # layout
    layout = Layout.from_yaml(config_dir / LAYOUT_NAME)

    # dem
    dem = stim.DetectorErrorModel.from_file(data_dir / f"{DEM_NAME}.dem")

    # sample simulated data
    seed = np.random.get_state()[1][0]  # current seed of numpy
    sampler = dem.compile_sampler(seed=seed)
    defects, log_flips, _ = sampler.sample(shots=NUM_SHOTS)

    # reshape and convert to xarray
    num_anc = len(layout.get_qubits(role="anc"))
    num_rounds = element["num_rounds"]
    defects = defects.reshape(NUM_SHOTS, num_rounds + 1, num_anc)
    defects, final_defects = defects[:, :-1, :], defects[:, -1, :]
    log_flips = log_flips[:, 0]

    ds = xr.Dataset(
        data_vars=dict(
            defects=(("shot", "qec_round", "anc_qubit"), defects.astype(bool)),
            final_defects=(("shot", "anc_qubit"), final_defects.astype(bool)),
            log_flips=(("shot",), log_flips.astype(bool)),
        ),
        coords=dict(
            shot=list(range(NUM_SHOTS)),
            qec_round=list(range(1, num_rounds + 1)),
            anc_qubit=layout.get_qubits(role="anc"),
        ),
    )
    ds.to_netcdf(data_dir / f"defects_{DEM_NAME}.nc")
