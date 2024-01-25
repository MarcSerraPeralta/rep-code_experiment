print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr

from qec_util import Layout
from iq_readout.two_state_classifiers import *
from rep_code.defects import to_defects
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230125_check_defect_rates_d5"

CONFIG_DATA = "config_data.yaml"
LAYOUT_NAME = "rep_code_layout_d5.yaml"
NOISE_MODEL = "ExperimentalNoiseModelExp"

###############################

print("Running script...")

with open(DATA_DIR / EXP_NAME / CONFIG_DATA, "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n" * 3, end="")  # for style purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)

    # load layout
    layout = Layout.from_yaml(config_dir / LAYOUT_NAME)
    proj_mat = layout.projection_matrix(stab_type="x_type")

    # process data
    dataset = xr.load_dataset(data_dir / f"measurements_{NOISE_MODEL}.nc")

    defects, final_defects, log_flips = to_defects(
        anc_meas=dataset.anc_meas,
        data_meas=dataset.data_meas,
        ideal_anc_meas=dataset.ideal_anc_meas,
        ideal_data_meas=dataset.ideal_data_meas,
        proj_mat=proj_mat,
    )

    ds = xr.Dataset(
        {
            "defects": defects.astype(bool),
            "final_defects": final_defects.astype(bool),
            "log_flips": log_flips.astype(bool),
        },
    )
    ds.to_netcdf(data_dir / f"defects_{NOISE_MODEL}.nc")

    num_rounds = element["num_rounds"]
    print("\033[F\033[K" * 3, end="", flush=True)
    print(f"defect rate (r={num_rounds}) {np.average(defects.values):0.4f}")
    print(f"final defect rate (r={num_rounds}) {np.average(final_defects.values):0.4f}")
    print(f"logical flips rate (r={num_rounds}) {np.average(log_flips.values):0.4f}")
