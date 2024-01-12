import pathlib
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit
from iq_readout.plots import plot_pdf_projected, plot_pdfs_projected

EXP_NAME = "distance3 01010"

EXP_DIR = pathlib.Path("processed") / EXP_NAME
FILE_NAMES = sorted(os.listdir(EXP_DIR))
CLASSIFIER = TwoStateLinearClassifierFit


for f_name in FILE_NAMES:
    cal_dir = EXP_DIR / f_name / "readout_calibration"

    if (cal_dir / f"readout_calibration_{CLASSIFIER.__name__}.npy").exists():
        continue

    # calibrate readout
    iq_readout_data = xr.load_dataset(cal_dir / "readout_calibration_iq.nc")
    calibration_params = {}

    for qubit in iq_readout_data.qubit:
        qubit_name = qubit.values.item()
        shots_0, shots_1 = (
            iq_readout_data.calibration.sel(state=i, qubit=qubit)
            .transpose("shot", "iq")
            .values
            for i in range(2)
        )
        classifier = CLASSIFIER().fit(shots_0, shots_1)
        cla_qubit_params = classifier.params()
        calibration_params[qubit_name] = cla_qubit_params

        fig, ax = plt.subplots()
        ax = plot_pdfs_projected(ax, shots_0, shots_1, classifier)
        fig.tight_layout()
        for format_ in ["pdf", "png"]:
            fig.savefig(
                cal_dir
                / f"readout_calibration_{CLASSIFIER.__name__}_{qubit_name}.{format_}",
                format=format_,
            )
        plt.close()

    np.save(
        cal_dir / f"readout_calibration_{CLASSIFIER.__name__}.npy",
        calibration_params,
    )
