import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit
from iq_readout.plots import plot_pdfs_projected
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3_s010"

CLASSIFIER = TwoStateLinearClassifierFit

#################################

cla_name = CLASSIFIER.__name__

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
STRING_DATA.pop("num_rounds")

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    print(
        f"\033[F\033[K{config_data['readout_calibration'].format(**element)}",
        flush=True,
    )
    cal_dir = DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)

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
        calibration_params[qubit_name] = classifier.params()

        # plot readout calibration
        fig, ax = plt.subplots()
        ax = plot_pdfs_projected(ax, shots_0, shots_1, classifier)
        fig.tight_layout()
        for format_ in ["pdf"]:
            fig.savefig(
                cal_dir / f"{cla_name}_{qubit_name}.{format_}",
                format=format_,
            )
        plt.close()

    np.save(cal_dir / f"{cla_name}_params.npy", calibration_params)
