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

EXP_NAME = "20230119_initial_data_d5_s01010"
COMB_NAME = "20230119_initial_data_d5_s01010_combined"

####################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

COMB_STRING_DATA = {
    k: v for k, v in config_data["string_data_options"].items() if k != "time"
}
new_config_data = deepcopy(config_data)
new_config_data["string_data_options"] = COMB_STRING_DATA
new_config_data["data"] = new_config_data["data"].replace("h{time}_", "")
new_config_data["config"] = new_config_data["config"].replace("h{time}_", "")
new_config_data["readout_calibration"] = new_config_data["readout_calibration"].replace(
    "h{time}_", ""
)
list_time = config_data["string_data_options"]["time"]

(DATA_DIR / COMB_NAME).mkdir(parents=True, exist_ok=True)
with open(DATA_DIR / COMB_NAME / "config_data.yaml", "w") as file:
    yaml.dump(new_config_data, file, default_flow_style=False)

print("\n", end="")  # for printing purposes

for element in sequence_generator(COMB_STRING_DATA):
    print(f"\033[F\033[K{new_config_data['data'].format(**element)}", flush=True)

    config_dir = (
        DATA_DIR / EXP_NAME / config_data["config"].format(**element, time=list_time[0])
    )
    new_config_dir = DATA_DIR / COMB_NAME / new_config_data["config"].format(**element)
    new_config_dir.mkdir(parents=True, exist_ok=True)

    # load & save layout and device characterization
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    layout.to_yaml(new_config_dir / "rep_code_layout.yaml")

    with open(config_dir / "device_characterization.yaml", "r") as file:
        device_characterization = yaml.safe_load(file)
    with open(config_dir / "device_characterization.yaml", "w") as file:
        yaml.dump(device_characterization, file, default_flow_style=False)

    # combine data
    list_data_qec = []
    list_data_cal = []
    for time in list_time:
        data_dir = (
            DATA_DIR / EXP_NAME / config_data["data"].format(**element, time=time)
        )
        data = xr.load_dataset(data_dir / "iq_data.nc")
        list_data_qec.append(data)

        data_dir = (
            DATA_DIR
            / EXP_NAME
            / config_data["readout_calibration"].format(**element, time=time)
        )
        data = xr.load_dataset(data_dir / "readout_calibration_iq.nc")
        list_data_cal.append(data)

    full_data_qec = xr.concat(list_data_qec, dim="shot")
    full_data_qec["shot"] = range(len(full_data_qec.shot))

    new_data_dir = DATA_DIR / COMB_NAME / new_config_data["data"].format(**element)
    new_data_dir.mkdir(parents=True, exist_ok=True)
    full_data_qec.to_netcdf(new_data_dir / "iq_data.nc")

    full_data_cal = xr.concat(list_data_cal, dim="shot")
    full_data_cal["shot"] = range(len(full_data_cal.shot))

    new_data_dir = (
        DATA_DIR / COMB_NAME / new_config_data["readout_calibration"].format(**element)
    )
    new_data_dir.mkdir(parents=True, exist_ok=True)
    full_data_cal.to_netcdf(new_data_dir / "readout_calibration_iq.nc")
