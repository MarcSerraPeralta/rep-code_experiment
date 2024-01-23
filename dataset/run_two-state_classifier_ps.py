import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from iq_readout.two_state_classifiers import *
from iq_readout.plots import plot_pdfs_projected
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

CLASSIFIER = DecayLinearClassifierFit

P0 = 0.5  # probability of the qubit being in state 0

#################################

cla_name = CLASSIFIER.__name__

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
STRING_DATA.pop("num_rounds")

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    cal_dir = DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)

    # calibrate readout
    iq_readout_data = xr.load_dataset(cal_dir / "readout_calibration_iq.nc")
    calibration_params_no_ps = {}
    calibration_params_ps = {}

    ps_fraction = []
    for qubit in iq_readout_data.qubit:
        # run first calibration
        qubit_name = qubit.values.item()
        shots_0, shots_1 = (
            iq_readout_data.calibration.sel(state=i, qubit=qubit)
            .transpose("shot", "iq")
            .values
            for i in range(2)
        )
        classifier = CLASSIFIER().fit(shots_0, shots_1)
        calibration_params_no_ps[qubit_name] = classifier.params()

        # plot readout calibration without PS
        fig, ax = plt.subplots()
        ax = plot_pdfs_projected(ax, shots_0, shots_1, classifier)
        fig.tight_layout()
        for format_ in ["pdf"]:
            fig.savefig(
                cal_dir / f"{cla_name}_{qubit_name}_no-ps.{format_}",
                format=format_,
            )
        plt.close()

        # run PS based on heralded init
        her_init_0, her_init_1 = (
            iq_readout_data.heralded_init.sel(state=i, qubit=qubit)
            .transpose("shot", "heralded_rep", "iq")
            .values
            for i in range(2)
        )
        her_init_0 = classifier.predict(her_init_0, p0=P0)
        her_init_1 = classifier.predict(her_init_1, p0=P0)
        mask_ps_0 = ~her_init_0.sum(axis=1).astype(bool)
        mask_ps_1 = ~her_init_1.sum(axis=1).astype(bool)
        shots_0_ps, shots_1_ps = shots_0[mask_ps_0], shots_1[mask_ps_1]
        ps_fraction.append(len(shots_0_ps) / len(shots_0))
        ps_fraction.append(len(shots_1_ps) / len(shots_1))

        # run second calibration
        classifier = CLASSIFIER().fit(shots_0_ps, shots_1_ps)
        calibration_params_ps[qubit_name] = classifier.params()

        # plot readout calibration without PS
        fig, ax = plt.subplots()
        ax = plot_pdfs_projected(ax, shots_0_ps, shots_1_ps, classifier)
        fig.tight_layout()
        for format_ in ["pdf"]:
            fig.savefig(
                cal_dir / f"{cla_name}_{qubit_name}_ps.{format_}",
                format=format_,
            )
        plt.close()

    np.save(cal_dir / f"{cla_name}_params_no-ps.npy", calibration_params_no_ps)
    np.save(cal_dir / f"{cla_name}_params_ps.npy", calibration_params_ps)

    print(
        f"\033[F\033[K{config_data['readout_calibration'].format(**element)} ps_fraction={np.average(ps_fraction)}",
        flush=True,
    )
