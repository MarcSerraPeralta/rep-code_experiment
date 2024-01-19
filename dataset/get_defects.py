import pathlib
import os

import numpy as np
import xarray as xr

from qec_util import Layout
from rep_code.defects import to_defects
from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit

EXP_NAME = "distance3_01010"

EXP_DIR = pathlib.Path("processed") / EXP_NAME
FILE_NAMES = sorted(next(os.walk(EXP_DIR))[1])
CLASSIFIER = TwoStateLinearClassifierFit
ROUNDS = list(range(1, 60 + 1))

for f_name in FILE_NAMES:
    run_dir = EXP_DIR / f_name
    qec_dir = run_dir / "qec_iq_data"
    cal_dir = run_dir / "readout_calibration"

    dec_dir = run_dir / "defects"
    dec_dir.mkdir(exist_ok=True, parents=True)

    # load classifier and layout
    layout = Layout.from_yaml(run_dir / "rep_code_layout.yaml")
    proj_mat = layout.projection_matrix(stab_type="x_type")

    cla_params = np.load(
        cal_dir / f"readout_calibration_{CLASSIFIER.__name__}.npy", allow_pickle=True
    ).item()
    classifiers = {q: CLASSIFIER().load(cla_params[q]) for q in layout.get_qubits()}

    ps_fraction = []
    for num_rounds in ROUNDS:
        dataset = xr.load_dataset(qec_dir / f"iq_data_r{num_rounds}.nc")

        defects, final_defects, log_flips = to_defects(
            dataset, proj_mat=proj_mat, classifiers=classifiers
        )

        ds = xr.Dataset(
            {
                "defects": defects.astype(bool),
                "final_defects": final_defects.astype(bool),
                "log_flips": log_flips.astype(bool),
            }
        )
        ds.to_netcdf(dec_dir / f"decoding_data_r{num_rounds}.nc")

        ps_fraction.append(len(defects.shot) / len(dataset.shot))

    print(f"PS heralded_init fraction (r={num_rounds}) {np.average(ps_fraction):0.3f}")
    print(f"defect rate (r={num_rounds}) {np.average(defects.values):0.4f}")
    print(f"final defect rate (r={num_rounds}) {np.average(final_defects.values):0.4f}")
    print(f"logical flips rate (r={num_rounds}) {np.average(log_flips.values):0.4f}")
