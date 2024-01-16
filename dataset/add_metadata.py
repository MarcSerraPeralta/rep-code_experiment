import pathlib
import os

import yaml

RAW_DIR = pathlib.Path("raw")

EXP_NAMES = next(os.walk(RAW_DIR))[1]  # list only directories

for exp_name in EXP_NAMES:
    run_names = next(os.walk(RAW_DIR / exp_name))[1]

    for run_name in run_names:
        run_dir = RAW_DIR / exp_name / run_name

        if exp_name == "distance3 01010":
            metadata = dict(
                anc_qubits=["Z1", "Z3"],
                data_qubits=["D4", "D5", "D7"],
                data_init={"D5": 0, "D4": 1, "D7": 0},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
            )
        elif exp_name == "distance3 10101":
            metadata = dict(
                anc_qubits=["Z1", "Z3"],
                data_qubits=["D4", "D5", "D7"],
                data_init={"D5": 1, "D4": 0, "D7": 1},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
            )
        elif exp_name == "distance5 01010":
            metadata = dict(
                anc_qubits=["Z1", "Z2", "Z3", "Z4"],
                data_qubits=["D3", "D4", "D5", "D6", "D7"],
                data_init={"D7": 0, "D4": 1, "D5": 0, "D6": 1, "D3": 0},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
            )
        elif exp_name == "distance5 10101":
            metadata = dict(
                anc_qubits=["Z1", "Z2", "Z3", "Z4"],
                data_qubits=["D3", "D4", "D5", "D6", "D7"],
                data_init={"D7": 1, "D4": 0, "D5": 1, "D6": 0, "D3": 1},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
            )
        else:
            raise ValueError(f"Experiment name does not have metadata: {exp_name}")

        with open(run_dir / "metadata.yaml", "w") as outfile:
            yaml.dump(metadata, outfile, default_flow_style=False)
