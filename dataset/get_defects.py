import pathlib
import os
import yaml

import numpy as np
import xarray as xr

from qec_util import Layout
from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit
from rep_code.defects import to_defects
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3_s010"

CLASSIFIER = TwoStateLinearClassifierFit
STRING_DATA = dict(
    num_rounds=list(range(1, 60 + 1)),
    time=[
        "092510",
        "034940",
        "080057",
        "070552",
        "054049",
        "024300",
        "044440",
        "101216",
        "030415",
        "032451",
        "041513",
        "063556",
        "095015",
        "073232",
        "085822",
        "021352",
        "014754",
        "060919",
        "051344",
        "083006",
    ],
    distance=3,
    state="010",
    data_qubits="547",
    basis="Z",
)

###############################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

print("\n" * 4, end="")  # for style purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    cal_dir = DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
    layout_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)

    # load classifier and layout
    layout = Layout.from_yaml(layout_dir / "rep_code_layout.yaml")
    proj_mat = layout.projection_matrix(stab_type="x_type")

    cla_params = np.load(
        cal_dir / f"{CLASSIFIER.__name__}_params.npy", allow_pickle=True
    ).item()
    classifiers = {q: CLASSIFIER().load(cla_params[q]) for q in layout.get_qubits()}

    # process data
    dataset = xr.load_dataset(data_dir / "iq_data.nc")

    defects, final_defects, log_flips = to_defects(
        dataset, proj_mat=proj_mat, classifiers=classifiers
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
    ds.to_netcdf(data_dir / f"defects_{CLASSIFIER.__name__}.nc")

    num_rounds = element["num_rounds"]
    print("\033[F\033[K" * 4, end="", flush=True)
    print(f"PS heralded_init fraction (r={num_rounds}) {ps_fraction:0.3f}")
    print(f"defect rate (r={num_rounds}) {np.average(defects.values):0.4f}")
    print(f"final defect rate (r={num_rounds}) {np.average(final_defects.values):0.4f}")
    print(f"logical flips rate (r={num_rounds}) {np.average(log_flips.values):0.4f}")
