import pathlib
import os
import yaml

import numpy as np
import xarray as xr

from qec_util import Layout
from surface_sim import Setup

from rep_code.circuits import memory_experiment
from rep_code.models import ExperimentalNoiseModelExp
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

####################

NOISE_NAME = "exp-circ-level_noise"

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)

    # load model and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    setup = Setup.from_yaml(config_dir / "device_characterization.yaml")
    qubit_inds = {q: layout.get_inds([q])[0] for q in layout.get_qubits()}
    model = ExperimentalNoiseModelExp(setup, qubit_inds)

    print(f"\033[F\033[K{config_data['data'].format(**element)}", flush=True)
    exp_circ = memory_experiment(
        model,
        layout,
        num_rounds=element["num_rounds"],
        data_init={
            f"D{q}": s for q, s in zip(element["data_qubits"], element["state"])
        },
        basis=element["basis"],
    )
    dem = exp_circ.detector_error_model()

    # store circuit and dem
    exp_circ.to_file(data_dir / f"{NOISE_NAME}.stim")
    dem.to_file(data_dir / f"{NOISE_NAME}.dem")
