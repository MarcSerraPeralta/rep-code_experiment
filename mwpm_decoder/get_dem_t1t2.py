import pathlib
import os
import yaml
from copy import deepcopy

import numpy as np
import xarray as xr

from qec_util import Layout
from surface_sim import Setup

from rep_code.circuits import memory_experiment
from rep_code.models import DecoherenceNoiseModelExp
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/dems"
)

EXP_NAME = "20230119_initial_data_d3"

####################

NOISE_NAME = "t1t2_noise"

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

list_states = config_data["string_data_options"].pop("state")

new_config_data = deepcopy(config_data)
new_config_data["data"] = new_config_data["data"].replace("_s{state}", "")
new_config_data["config"] = new_config_data["config"].replace("_s{state}", "")
STRING_DATA = new_config_data["string_data_options"]

# the DEM used for decoding should not depend on the initial state,
# thus we select one of the states to build the DEM
state = sorted(list_states)[0]

(OUTPUT_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / EXP_NAME / "config_data.yaml", "w") as file:
    yaml.dump(new_config_data, file, default_flow_style=False)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = (
        DATA_DIR / EXP_NAME / config_data["config"].format(**element, state=state)
    )
    output_dir = OUTPUT_DIR / EXP_NAME / new_config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\033[F\033[K{output_dir}", flush=True)

    # load model and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    setup = Setup.from_yaml(config_dir / "device_characterization.yaml")
    qubit_inds = {q: layout.get_inds([q])[0] for q in layout.get_qubits()}
    model = DecoherenceNoiseModelExp(setup, qubit_inds, symmetric_noise=True)

    exp_circ = memory_experiment(
        model,
        layout,
        num_rounds=element["num_rounds"],
        data_init={f"D{q}": s for q, s in zip(element["data_qubits"], state)},
        basis=element["basis"],
    )
    dem = exp_circ.detector_error_model()

    # store circuit, dem, layout, setup
    exp_circ.to_file(output_dir / f"{NOISE_NAME}.stim")
    dem.to_file(output_dir / f"{NOISE_NAME}.dem")

    out_conf_dir = OUTPUT_DIR / EXP_NAME / new_config_data["config"].format(**element)
    out_conf_dir.mkdir(parents=True, exist_ok=True)
    layout.to_yaml(out_conf_dir / "rep_code_layout.yaml")
    setup.to_yaml(out_conf_dir / "device_characterization.yaml")
