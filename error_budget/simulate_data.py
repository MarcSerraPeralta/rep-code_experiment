print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr

from qec_util import Layout
from surface_sim import Setup
from rep_code.dataset import sequence_generator
from rep_code.circuits import memory_experiment
from rep_code.models import ExperimentalNoiseModelExp

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

EXP_NAME = "20230123_error_budget_simulation_d5"

CONFIG_DATA = "config_data_scan_cz_error_prob.yaml"
NUM_SHOTS = 100_000

LAYOUT_NAME = "rep_code_layout_d5.yaml"
NOISE_PARAMS_NAME = "circ_level_noise.yaml"

###############################

print("Running script...")

CONFIG_DIR = pathlib.Path("configs")
(DATA_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)

with open(CONFIG_DIR / CONFIG_DATA, "r") as file:
    config_data = yaml.safe_load(file)
with open(DATA_DIR / EXP_NAME / CONFIG_DATA, "w") as file:
    yaml.dump(config_data, file, default_flow_style=False)

STRING_DATA = config_data["string_data_options"]

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"{config_data['data'].format(**element)}", end="\r")

    # layout
    layout = Layout.from_yaml(CONFIG_DIR / LAYOUT_NAME)
    layout.to_yaml(config_dir / LAYOUT_NAME)

    # noise
    with open(CONFIG_DIR / NOISE_PARAMS_NAME, "r") as file:
        device_characterization = yaml.safe_load(file)
    with open(config_dir / NOISE_PARAMS_NAME, "w") as file:
        yaml.dump(device_characterization, file, default_flow_style=False)

    setup = Setup.from_yaml(config_dir / NOISE_PARAMS_NAME)
    for param in setup.free_params:
        setup.set_var_param(param, element[param])

    qubit_inds = {q: layout.get_inds([q])[0] for q in layout.get_qubits()}
    model = ExperimentalNoiseModelExp(setup, qubit_inds)
    NOISE_MODEL = type(model).__name__

    # circuit and dem
    data_init = {f"D{q}": s for q, s in zip(element["data_qubits"], element["state"])}
    circuit = memory_experiment(
        model,
        layout,
        num_rounds=element["num_rounds"],
        data_init=data_init,
        basis=element["basis"],
    )
    dem = circuit.detector_error_model()

    circuit.to_file(data_dir / f"circuit_{NOISE_MODEL}.stim")
    dem.to_file(data_dir / f"dem_{NOISE_MODEL}.dem")

    # sample simulated data
    seed = np.random.get_state()[1][0]  # current seed of numpy
    sampler = circuit.compile_sampler(seed=seed)
    meas = sampler.sample(shots=NUM_SHOTS)
    ideal_sampler = circuit.without_noise().compile_sampler()
    ideal_meas = ideal_sampler.sample(shots=1)[0]

    # reshape and convert to xarray
    num_data = len(layout.get_qubits(role="data"))
    num_anc = len(layout.get_qubits(role="anc"))
    num_rounds = element["num_rounds"]

    anc_meas, data_meas = meas[:, :-num_data], meas[:, -num_data:]
    anc_meas = anc_meas.reshape(NUM_SHOTS, num_rounds, num_anc)

    ideal_anc_meas, ideal_data_meas = ideal_meas[:-num_data], ideal_meas[-num_data:]
    ideal_anc_meas = ideal_anc_meas.reshape(num_rounds, num_anc)

    data_init = np.array([data_init[q] for q in layout.get_qubits(role="data")])

    measurements = xr.Dataset(
        data_vars=dict(
            anc_meas=(("shot", "qec_round", "anc_qubit"), anc_meas.astype(bool)),
            data_meas=(("shot", "data_qubit"), data_meas.astype(bool)),
            data_init=(
                ("data_qubit"),
                data_init.astype(bool),
            ),
            ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas.astype(bool)),
            ideal_data_meas=(("data_qubit"), ideal_data_meas.astype(bool)),
        ),
        coords=dict(
            shot=list(range(NUM_SHOTS)),
            anc_qubit=layout.get_qubits(role="anc"),
            data_qubit=layout.get_qubits(role="data"),
            qec_round=list(range(1, num_rounds + 1)),
            meas_reset=False,
            rot_basis=True if element["basis"] == "X" else False,
        ),
    )
    measurements.to_netcdf(data_dir / f"measurements_{NOISE_MODEL}.nc")
