print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import pymatching
import stim

from qec_util import Layout
from rep_code.defects import get_defect_vector
from rep_code.dataset import sequence_generator


DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/error_budget"
)

EXP_NAME = "20230123_error_budget_simulation_d5"

CONFIG_DATA = "config_data_scan_cz_error_prob.yaml"
LAYOUT_NAME = "rep_code_layout_d5.yaml"
NOISE_MODEL = "ExperimentalNoiseModelExp"

####################

print("Running script...")

with open(DATA_DIR / EXP_NAME / CONFIG_DATA, "r") as file:
    config_data = yaml.safe_load(file)
with open(OUTPUT_DIR / EXP_NAME / CONFIG_DATA, "w") as file:
    yaml.dump(config_data, file, default_flow_style=False)

STRING_DATA = config_data["string_data_options"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)

    # load
    layout = Layout.from_yaml(config_dir / LAYOUT_NAME)

    defects_xr = xr.load_dataset(data_dir / f"defects_{NOISE_MODEL}.nc")
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

    # dem and mwpm
    dem = stim.DetectorErrorModel.from_file(data_dir / f"dem_{NOISE_MODEL}.dem")
    mwpm = pymatching.Matching(dem)

    # decode
    prediction = mwpm.decode_batch(defect_vec)
    # Note: if not flatten, the log_errors is incorrect
    log_errors = log_flips != prediction.flatten()

    # store logical errors
    log_errors = xr.DataArray(data=log_errors, coords=dict(shot=defects_xr.shot.values))

    output_dir.mkdir(parents=True, exist_ok=True)
    log_errors.to_netcdf(output_dir / f"log_err_{NOISE_MODEL}.nc")

    print(
        f"\033[F\033[K{config_data['data'].format(**element)} p_L={np.average(log_errors):0.3f}",
        flush=True,
    )
