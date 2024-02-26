print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr

from qec_util import Layout
from iq_readout.three_state_classifiers import *
from rep_code.defects import to_defects, get_measurements, ps_shots_heralded
from rep_code.defects.analysis import get_three_state_probs
from rep_code.dataset import sequence_generator

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230119_initial_data_d3"
IQ_DATA_NAME = "iq_data"
PROBS_NAME = "state_probs"  # will include the classifier name
CLASSIFIER = GaussMixClassifier

###############################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

print("\n", end="")  # for style purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    cal_dir = DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    print(f"\033[F\033[K{data_dir}", flush=True)

    # load classifier and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    num_anc = len(layout.get_qubits(role="anc"))
    num_data = len(layout.get_qubits(role="data"))

    cla_name = CLASSIFIER.__name__
    cla_params = np.load(
        cal_dir / f"3state_{cla_name}_params_ps.npy", allow_pickle=True
    ).item()
    classifiers = {q: CLASSIFIER().load(cla_params[q]) for q in layout.get_qubits()}

    # process data
    dataset = xr.load_dataset(data_dir / f"{IQ_DATA_NAME}.nc")
    anc_meas, data_meas = dataset.anc_meas, dataset.data_meas

    # digitize measurements
    _, _, heralded_init = get_measurements(dataset, classifiers)

    # post select based on heralded measurement
    shots = ps_shots_heralded(heralded_init)
    anc_meas, data_meas = anc_meas.sel(shot=shots), data_meas.sel(shot=shots)

    # get state probabilities for ancillas
    probs_anc = np.zeros((num_anc, len(dataset.qec_round), 3))

    for i, qec_round in enumerate(dataset.qec_round):
        for j, anc_qubit in enumerate(layout.get_qubits(role="anc")):
            classifier = classifiers[anc_qubit]
            iq_data = anc_meas.sel(
                anc_qubit=anc_qubit,
                qec_round=qec_round,
            ).transpose("shot", "iq")
            p0, p1, p2 = get_three_state_probs(classifier, iq_data)
            probs_anc[j, i] = np.array([p0, p1, p2])

    # get state probabilities for data qubits
    probs_data = np.zeros((num_data, 3))
    for j, data_qubit in enumerate(layout.get_qubits(role="data")):
        classifier = classifiers[data_qubit]
        iq_data = data_meas.sel(data_qubit=data_qubit).transpose("shot", "iq")
        p0, p1, p2 = get_three_state_probs(classifier, iq_data)
        probs_data[j] = np.array([p0, p1, p2])

    ds = xr.Dataset(
        data_vars=dict(
            probs_anc=(("anc_qubit", "qec_round", "state"), probs_anc),
            probs_data=(("data_qubit", "state"), probs_data),
        ),
        coords=dict(
            state=[0, 1, 2],
            anc_qubit=layout.get_qubits(role="anc"),
            data_qubit=layout.get_qubits(role="data"),
            qec_round=dataset.qec_round,
        ),
    )
    ds.to_netcdf(data_dir / f"{PROBS_NAME}_{cla_name}.nc")
