import pathlib
import os

import yaml

RAW_DATA_DIR = pathlib.Path("/tudelft.net/staff-umbrella/repcode/")

EXP_NAMES = next(os.walk(RAW_DATA_DIR))[1]  # list only directories

for exp_name in EXP_NAMES:
    run_names = next(os.walk(RAW_DATA_DIR / exp_name))[1]

    for run_name in run_names:
        run_dir = RAW_DATA_DIR / exp_name / run_name

        if exp_name == "distance3_01010":
            metadata = dict(
                anc_qubits=["Z1", "Z3"],
                data_qubits=["D5", "D4", "D7"],
                data_init={"D5": 0, "D4": 1, "D7": 0},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
                distance=3,
            )
        elif exp_name == "distance3_10101":
            metadata = dict(
                anc_qubits=["Z1", "Z3"],
                data_qubits=["D5", "D4", "D7"],
                data_init={"D5": 1, "D4": 0, "D7": 1},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
                distance=3,
            )
        elif exp_name == "distance5_01010":
            metadata = dict(
                anc_qubits=["Z1", "Z2", "Z3", "Z4"],
                data_qubits=["D7", "D4", "D5", "D6", "D3"],
                data_init={"D7": 0, "D4": 1, "D5": 0, "D6": 1, "D3": 0},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
                distance=5,
            )
        elif exp_name == "distance5_10101":
            metadata = dict(
                anc_qubits=["Z1", "Z2", "Z3", "Z4"],
                data_qubits=["D7", "D4", "D5", "D6", "D3"],
                data_init={"D7": 1, "D4": 0, "D5": 1, "D6": 0, "D3": 1},
                num_rounds=list(range(1, 60 + 1)),
                rot_basis=False,
                readout_calibration_states=[0, 1, 2],
                distance=5,
            )
        else:
            raise ValueError(f"Experiment name does not have metadata: {exp_name}")

        with open(run_dir / "metadata.yaml", "w") as outfile:
            yaml.dump(metadata, outfile, default_flow_style=False)
