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
NN_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/nn_data"
)

EXP_NAME = "20230119_initial_data_d3"

TEST = 25_000
TRAIN = 0.9  # fraction of the remaining shots

####################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n", end="")  # for printing purposes

SEED = 11666

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    read_dir = (
        DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
    )
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)

    nn_config_dir = NN_DIR / EXP_NAME / "config"
    nn_config_dir.mkdir(parents=True, exist_ok=True)
    nn_read_dir = NN_DIR / EXP_NAME / "readout_calibration"
    nn_read_dir.mkdir(parents=True, exist_ok=True)

    print(f"\033[F\033[K{data_dir}", flush=True)

    # copy layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    layout.to_yaml(nn_config_dir / "layout.yaml")

    # copy readout calibration data
    readout_files = [f for f in os.listdir(read_dir) if (".nc" in f) or (".npy" in f)]
    for file in readout_files:
        shutil.copy(read_dir / file, nn_read_dir)

    # load and split IQ data
    iq_data = xr.load_dataset(data_dir / f"iq_data.nc")
    shots = np.arange(len(defects.shot), dtype=int)
    np.random.seed(SEED)
    np.random.shuffle(shots)
    num_train = int((len(shots) - TEST) * TRAIN)
    num_test = TEST
    test = iq_data.sel(shot=shots[:num_test])
    train = iq_data.sel(shot=shots[num_test : num_test + num_train])
    val = iq_data.sel(shot=shots[num_test + num_train :])

    # store train, val, test
    test_dir = NN_DIR / EXP_NAME / "test" / config_data["data"].format(**element)
    test_dir.mkdir(parents=True, exist_ok=True)
    test.to_netcdf(test_dir / "measurements.nc")

    train_dir = NN_DIR / EXP_NAME / "train" / config_data["data"].format(**element)
    train_dir.mkdir(parents=True, exist_ok=True)
    train.to_netcdf(train_dir / "measurements.nc")

    val_dir = NN_DIR / EXP_NAME / "val" / config_data["data"].format(**element)
    val_dir.mkdir(parents=True, exist_ok=True)
    val.to_netcdf(val_dir / "measurements.nc")
