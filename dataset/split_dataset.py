import pathlib
import yaml
import os
import shutil
from copy import deepcopy

import numpy as np
import xarray as xr

from qec_util import Layout
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
SPLIT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/nn_data"
)

EXP_NAME = "20230119_initial_data_d3"

SPLITS = {
    "test": 25_000,
    "train": 0.9,
    "val": None,
}

####################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)
with open(SPLIT_DIR / EXP_NAME / "config_data.yaml", "w") as file:
    yaml.dump(config_data, file, default_flow_style=False)

STRING_DATA = config_data["string_data_options"]

SEED = 11666

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    read_dir = (
        DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
    )
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)

    split_config_dir = SPLIT_DIR / EXP_NAME / "config"
    split_config_dir.mkdir(parents=True, exist_ok=True)
    split_read_dir = SPLIT_DIR / EXP_NAME / "readout_calibration"
    split_read_dir.mkdir(parents=True, exist_ok=True)
    split_data_dir = SPLIT_DIR / EXP_NAME / config_data["data"].format(**element)
    split_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\033[F\033[K{data_dir}", flush=True)

    # copy layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    layout.to_yaml(split_config_dir / "layout.yaml")

    # copy readout calibration data
    readout_files = [f for f in os.listdir(read_dir) if (".nc" in f) or (".npy" in f)]
    for file in readout_files:
        shutil.copy(read_dir / file, split_read_dir)

    # load IQ data
    iq_data = xr.load_dataset(data_dir / f"iq_data.nc")

    # split data
    shots = np.arange(len(defects.shot), dtype=int)
    np.random.seed(SEED)
    np.random.shuffle(shots)

    for name, n_shots in SPLITS.items():
        if isinstance(n_shots, float):
            n_shots = int(n_shots * len(shots))
        elif n_shots is None:
            n_shots = len(shots)

        # works as array.pop(), which does not exist
        current, shots = shots[:n_shots], shots[n_shots:]
        split_ds = iq_data.sel(shot=current)
        split_ds.to_netcdf(split_data_dir / f"iq_data_{name}.nc")

    if len(shots) != 0:
        print(
            f"\033[F\033[KWARNING: not all shots have been used for {data_dir}\n",
            flush=True,
        )
