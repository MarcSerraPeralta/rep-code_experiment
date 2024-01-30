print("Importing libraries...")
import pathlib
import os
import yaml
from copy import deepcopy

import numpy as np
import xarray as xr
import stim

from qec_util import Layout
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAMES = {
    "01010": "20230119_initial_data_d5_s01010_combined",
    "10101": "20230119_initial_data_d5_s10101_combined",
}
COMB_NAME = "20230119_initial_data_d5"

####################

print("Running script...")

list_states = list(EXP_NAMES.keys())
list_dirs = [EXP_NAMES[s] for s in list_states]

with open(DATA_DIR / list_dirs[0] / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

for exp_name in list_dirs[1:]:
    with open(DATA_DIR / exp_name / "config_data.yaml", "r") as file:
        other_config_data = yaml.safe_load(file)

    # check that they can be merged
    assert (
        other_config_data["string_data_options"] == config_data["string_data_options"]
    )

new_config_data = deepcopy(config_data)
new_config_data["string_data_options"]["state"] = list_states
new_config_data["data"] = new_config_data["data"].replace(
    "_r{num_rounds}", "_s{state}_r{num_rounds}"
)

(DATA_DIR / COMB_NAME).mkdir(parents=True, exist_ok=True)
with open(DATA_DIR / COMB_NAME / "config_data.yaml", "w") as file:
    yaml.dump(new_config_data, file, default_flow_style=False)

print("\n", end="")  # for printing purposes

for element in sequence_generator(config_data["string_data_options"]):
    print(f"\033[F\033[K{new_config_data['data'].format(**element)}", flush=True)

    config_dir = (
        DATA_DIR
        / list_dirs[0]
        / config_data["config"].format(**element, time=list_time[0])
    )
    new_config_dir = DATA_DIR / COMB_NAME / new_config_data["config"].format(**element)
    new_config_dir.mkdir(parents=True, exist_ok=True)

    # load & save layout and device characterization
    # ASSUMES THE LAYOUTS AND DEVICE CHARACTERIZATION ARE THE SAME
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    layout.to_yaml(new_config_dir / "rep_code_layout.yaml")

    with open(config_dir / "device_characterization.yaml", "r") as file:
        device_characterization = yaml.safe_load(file)
    with open(new_config_dir / "device_characterization.yaml", "w") as file:
        yaml.dump(device_characterization, file, default_flow_style=False)

    # combine readout data and create folder for each state
    list_data_cal = []
    for state, exp_name in EXP_NAMES.items():
        data_dir = DATA_DIR / exp_name / config_data["data"].format(**element)
        new_data_dir = (
            DATA_DIR
            / COMB_NAME
            / new_config_data["data"].format(**element, state=state)
        )
        new_data_dir.mkdir(parents=True, exist_ok=True)

        data = xr.load_dataset(data_dir / "iq_data.nc")
        data.to_netcdf(new_data_dir / "iq_data.nc")

        data_dir = (
            DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
        )
        data = xr.load_dataset(data_dir / "readout_calibration_iq.nc")
        list_data_cal.append(data)

    full_data_cal = xr.concat(list_data_cal, dim="shot")
    full_data_cal["shot"] = range(len(full_data_cal.shot))

    new_data_dir = (
        DATA_DIR / COMB_NAME / new_config_data["readout_calibration"].format(**element)
    )
    new_data_dir.mkdir(parents=True, exist_ok=True)
    full_data_cal.to_netcdf(new_data_dir / "readout_calibration_iq.nc")
