print("Importing libraries...")
import pathlib
import os
from copy import deepcopy
import yaml
import re

import numpy as np
import xarray as xr

from qce_interp import DataManager, QubitIDObj, ParityType, Surface17Layer, StateKey
from rep_code.layout import get_rep_code_layout
from rep_code.dataset import calibration_to_xarray, qec_to_xarray


RAW_DATA_DIR = pathlib.Path("/tudelft.net/staff-umbrella/repcode/")
PRO_DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

RAW_EXP_NAME = "distance5_01010"
PRO_EXP_NAME = "20230119_initial_data_d5_s01010"

STRING_FORMAT = dict(
    data="rep-code_d{distance}_s{state}_q{data_qubits}_b{basis}_h{time}_r{num_rounds}",
    config="rep-code_d{distance}_s{state}_q{data_qubits}_b{basis}_h{time}_config",
    readout_calibration="rep-code_d{distance}_s{state}_q{data_qubits}_b{basis}_h{time}_readout_calibration",
)

RUN_DIRS = sorted(os.listdir(RAW_DATA_DIR / RAW_EXP_NAME))

#####################################

RAW_EXP_DIR = RAW_DATA_DIR / RAW_EXP_NAME
PRO_EXP_DIR = PRO_DATA_DIR / PRO_EXP_NAME
PRO_EXP_DIR.mkdir(parents=True, exist_ok=True)

# find all elements that appear inbetween curly brackets
string_data_options = {k: [] for k in re.findall(r"\{([^}]+)\}", STRING_FORMAT["data"])}

for k, run_dir in enumerate(RUN_DIRS):
    print(f"\n\n{k+1}/{len(RUN_DIRS)} {run_dir}")

    # load metadata and data
    with open(RAW_EXP_DIR / run_dir / "metadata.yaml", "r") as file:
        metadata = yaml.safe_load(file)

    # prepare string format data
    # skip D in label of data_qubits
    string_data = deepcopy(metadata)
    string_data["state"] = "".join(
        map(str, [metadata["data_init"][q] for q in metadata["data_qubits"]])
    )
    string_data["data_qubits"] = "".join([q[1:] for q in metadata["data_qubits"]])
    string_data["time"] = run_dir.split("_")[0]
    string_data["basis"] = "X" if metadata["rot_basis"] else "Z"

    print("Loading DataManager...")
    raw_data_file = RAW_EXP_DIR / run_dir / f"{run_dir}.hdf5"
    data_manager = DataManager.from_file_path(
        file_path=raw_data_file,
        rounds=metadata["num_rounds"],
        heralded_initialization=True,
        qutrit_calibration_points=True,
        involved_data_qubit_ids=[QubitIDObj(q) for q in metadata["data_qubits"]],
        involved_ancilla_qubit_ids=[QubitIDObj(q) for q in metadata["anc_qubits"]],
        expected_parity_lookup={
            QubitIDObj(q): ParityType.ODD for q in metadata["data_qubits"]
        },
        device_layout=Surface17Layer(),
    )

    # STORE PROCESSED INFORMATION
    config_path = PRO_EXP_DIR / STRING_FORMAT["config"].format(**string_data)
    config_path.mkdir(parents=True, exist_ok=True)

    # layout
    layout = get_rep_code_layout(metadata["data_qubits"] + metadata["anc_qubits"])
    layout.to_yaml(config_path / "rep_code_layout.yaml")

    # device noise
    with open(RAW_EXP_DIR / run_dir / "device_characterization.yaml", "r") as file:
        device_characterization = yaml.safe_load(file)
    with open(config_path / "device_characterization.yaml", "w") as file:
        yaml.dump(device_characterization, file, default_flow_style=False)

    # readout calibration data
    print("\nConverting readout calibration data to xarray...")
    calibration_ds = calibration_to_xarray(
        data_manager=data_manager,
        calibration_states=metadata["readout_calibration_states"],
        qubits=metadata["data_qubits"] + metadata["anc_qubits"],
    )
    cal_dir = PRO_EXP_DIR / STRING_FORMAT["readout_calibration"].format(**string_data)
    cal_dir.mkdir(exist_ok=True, parents=True)
    calibration_ds.to_netcdf(cal_dir / "readout_calibration_iq.nc")

    # QEC cycle data
    print("\nConverting QEC data to xarray...")
    for k, num_rounds in enumerate(metadata["num_rounds"]):
        print(f"{k+1}/{len(metadata['num_rounds'])}\r", end="")
        qec_ds = qec_to_xarray(
            data_manager=data_manager,
            data_qubits=layout.get_qubits(role="data"),
            anc_qubits=layout.get_qubits(role="anc"),
            num_rounds=num_rounds,
            data_init=metadata["data_init"],
            proj_matrix=layout.projection_matrix(stab_type="x_type"),
        )
        qec_ds = qec_ds.assign_coords(
            meas_reset=False,
            rot_basis=metadata["rot_basis"],
        )

        string_data["num_rounds"] = num_rounds
        qec_dir = PRO_EXP_DIR / STRING_FORMAT["data"].format(**string_data)
        qec_dir.mkdir(exist_ok=True, parents=True)
        qec_ds.to_netcdf(qec_dir / f"iq_data.nc")

        # update string data options
        for k in string_data_options:
            if string_data[k] not in string_data_options[k]:
                string_data_options[k].append(string_data[k])

    print("\nDone!")

# store string format
STRING_FORMAT["string_data_options"] = string_data_options
with open(PRO_EXP_DIR / "config_data.yaml", "w") as file:
    yaml.dump(STRING_FORMAT, file, default_flow_style=False)
