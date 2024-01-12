import pathlib
import os

import numpy as np
import xarray as xr

from qec_util import Layout
from rep_code.defects import (
    get_measurements,
    ps_shots_heralded,
    get_defects,
    get_syndromes,
    get_final_defects,
)
from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit

EXP_NAME = "distance3 01010"

EXP_DIR = pathlib.Path("processed") / EXP_NAME
FILE_NAMES = sorted(os.listdir(EXP_DIR))
CLASSIFIER = TwoStateLinearClassifierFit
ROUNDS = list(range(1, 60 + 1))

for f_name in FILE_NAMES:
    run_dir = EXP_DIR / f_name
    qec_dir = run_dir / "qec_iq_data"
    cal_dir = run_dir / "readout_calibration"

    # load classifier and layout
    cla_params = np.load(
        cal_dir / f"readout_calibration_{CLASSIFIER.__name__}.npy", allow_pickle=True
    ).item()
    layout = Layout.from_yaml(run_dir / "rep_code_layout.yaml")

    dec_dir = run_dir / "decoding_data"
    dec_dir.mkdir(exist_ok=True, parents=True)

    ps_fraction = []
    for num_rounds in ROUNDS:
        dataset = xr.load_dataset(qec_dir / f"iq_data_r{num_rounds}.nc")

        # digitize measurements
        anc_meas, data_meas, heralded_init = get_measurements(
            dataset, CLASSIFIER(), cla_params
        )

        # post select based on heralded measurement
        shots = ps_shots_heralded(heralded_init)
        anc_meas, data_meas = anc_meas.sel(shot=shots), data_meas.sel(shot=shots)
        ps_fraction.append(len(shots) / len(dataset.shot) * 100)

        # compute defects
        proj_mat = layout.projection_matrix(stab_type="x_type")
        anc_flips = anc_meas ^ dataset.ideal_anc_meas
        data_flips = data_meas ^ dataset.ideal_data_meas
        # the initial frame is already present in the ideal_anc_meas

        syndromes = get_syndromes(anc_flips)
        defects = get_defects(syndromes)

        proj_syndrome = (data_flips @ proj_mat) % 2
        final_defects = get_final_defects(syndromes, proj_syndrome)

        log_errors = data_flips.sum(dim="data_qubit") % 2

        ds = xr.Dataset(
            {
                "defects": defects.astype(bool),
                "final_defects": final_defects.astype(bool),
                "log_errors": log_errors.astype(bool),
            }
        )
        ds.to_netcdf(dec_dir / f"decoding_data_r{num_rounds}.nc")

    print(f"PS heralded_init fraction {np.average(ps_fraction):0.2f}")
    print("defect average", np.average(defects.values))
    print("final defect average", np.average(final_defects.values))
    print("logical flips average", np.average(log_errors.values))
