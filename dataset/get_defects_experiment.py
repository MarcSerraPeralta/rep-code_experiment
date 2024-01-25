print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr

from qec_util import Layout
from iq_readout.two_state_classifiers import *
from rep_code.defects import to_defects, get_measurements
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

CLASSIFIER = DecayLinearClassifierFit

###############################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n" * 4, end="")  # for style purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    cal_dir = DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)

    # load classifier and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    proj_mat = layout.projection_matrix(stab_type="x_type")

    cla_name = CLASSIFIER.__name__
    cla_params = np.load(
        cal_dir / f"{cla_name}_params_ps.npy", allow_pickle=True
    ).item()
    classifiers = {q: CLASSIFIER().load(cla_params[q]) for q in layout.get_qubits()}

    # process data
    dataset = xr.load_dataset(data_dir / "iq_data.nc")

    # digitize measurements
    anc_meas, data_meas, heralded_init = get_measurements(dataset, classifiers)

    # post select based on heralded measurement
    shots = ps_shots_heralded(heralded_init)
    anc_meas, data_meas = anc_meas.sel(shot=shots), data_meas.sel(shot=shots)

    defects, final_defects, log_flips = to_defects(
        anc_meas=anc_meas,
        data_meas=data_meas,
        ideal_anc_meas=dataset.ideal_anc_meas,
        ideal_data_meas=dataset.ideal_data_meas,
        proj_mat=proj_mat,
    )
    ps_fraction = len(defects.shot) / len(dataset.shot)

    ds = xr.Dataset(
        {
            "defects": defects.astype(bool),
            "final_defects": final_defects.astype(bool),
            "log_flips": log_flips.astype(bool),
        },
        attrs=dict(ps_fraction=ps_fraction),
    )
    ds.to_netcdf(data_dir / f"defects_{cla_name}.nc")

    num_rounds = element["num_rounds"]
    print("\033[F\033[K" * 4, end="", flush=True)
    print(f"PS heralded_init fraction (r={num_rounds}) {ps_fraction:0.3f}")
    print(f"defect rate (r={num_rounds}) {np.average(defects.values):0.4f}")
    print(f"final defect rate (r={num_rounds}) {np.average(final_defects.values):0.4f}")
    print(f"logical flips rate (r={num_rounds}) {np.average(log_flips.values):0.4f}")
