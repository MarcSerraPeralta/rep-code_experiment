print("Importing libraries...")
import pathlib
import os
from copy import deepcopy

import yaml
import numpy as np
import xarray as xr

from qce_interp import DataManager, QubitIDObj, ParityType, Surface17Layer, StateKey
from rep_code.layout import get_rep_code_layout

EXP_NAME = "distance3 01010"

RAW_DIR = pathlib.Path("raw")
PRO_DIR = pathlib.Path("processed")
(PRO_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)

RUN_DIRS_ALL = sorted(os.listdir(RAW_DIR / EXP_NAME))
RUN_DIRS = [d for d in RUN_DIRS_ALL if d not in os.listdir(PRO_DIR / EXP_NAME)]

for k, run_dir in enumerate(RUN_DIRS):
    print(f"\n\n {k+1}/{len(RUN_DIRS)} {run_dir}")

    # load metadata
    with open(RAW_DIR / EXP_NAME / run_dir / "metadata.yaml", "r") as file:
        metadata = yaml.safe_load(file)

    list_num_rounds = metadata["num_rounds"]
    all_qubits = metadata["data_qubits"] + metadata["anc_qubits"]

    # load data
    print("Loading DataManager...")
    raw_data_file = RAW_DIR / EXP_NAME / run_dir / f"{run_dir}.hdf5"
    data_manager = DataManager.from_file_path(
        file_path=raw_data_file,
        rounds=list_num_rounds,
        heralded_initialization=True,
        qutrit_calibration_points=True,
        involved_data_qubit_ids=[QubitIDObj(q) for q in metadata["data_qubits"]],
        involved_ancilla_qubit_ids=[QubitIDObj(q) for q in metadata["anc_qubits"]],
        expected_parity_lookup={
            QubitIDObj(q): ParityType.ODD for q in metadata["data_qubits"]
        },
        device_layout=Surface17Layer(),
    )
    idx_kernel = data_manager._experiment_index_kernel

    # STORE PROCESSED INFORMATION
    out_run_dir = PRO_DIR / EXP_NAME / run_dir
    out_run_dir.mkdir(exist_ok=True, parents=True)

    # layout file
    layout = get_rep_code_layout(all_qubits)
    layout.to_yaml(out_run_dir / "rep_code_layout.yaml")

    # calibration data
    print("Converting calibration data to xarray...")

    meas_cal, heral_cal = [], []

    for qubit in all_qubits:
        print(f"\rqubit={qubit}", end="")
        raw_shots = data_manager.get_state_classifier(QubitIDObj(qubit)).shots
        meas_cal_qubit, heral_cal_qubit = [], []
        for state in metadata["readout_calibration_states"]:
            print(f"\rqubit={qubit} state={state}", end="")
            meas_cal_idx = idx_kernel.get_projected_calibration_acquisition_indices(
                qubit_id=QubitIDObj(qubit), state=StateKey(state)
            )
            meas_cal_data = raw_shots[meas_cal_idx]
            meas_cal_qubit.append([meas_cal_data.real, meas_cal_data.imag])
            heral_cal_idx = idx_kernel.get_heralded_calibration_acquisition_indices(
                qubit_id=QubitIDObj(qubit), state=StateKey(state)
            )
            heral_cal_data = raw_shots[heral_cal_idx]
            heral_cal_qubit.append([heral_cal_data.real, heral_cal_data.imag])
        meas_cal.append(meas_cal_qubit)
        heral_cal.append(heral_cal_qubit)

    heralded_reps = 1 if len(heral_cal_data.shape) == 1 else heral_cal_data.shape[1]
    heralded_shots = heral_cal_data.shape[0]

    ds = xr.Dataset(
        data_vars=dict(
            calibration=(("qubit", "state", "iq", "shot"), meas_cal),
            heralded_init=(("qubit", "state", "iq", "shot"), heral_cal),
        ),
        coords=dict(
            state=metadata["readout_calibration_states"],
            qubit=all_qubits,
            shot=list(range(heralded_shots)),
            heralded_rep=list(range(heralded_reps)),
            iq=["I", "Q"],
        ),
    )
    cal_dir = out_run_dir / "readout_calibration"
    cal_dir.mkdir(exist_ok=True, parents=True)
    ds.to_netcdf(cal_dir / "readout_calibration_iq.nc")

    # QEC cycle data
    print("\nConverting QEC data to xarray...")
    qec_dir = out_run_dir / "qec_iq_data"
    qec_dir.mkdir(exist_ok=True, parents=True)

    for num_rounds in list_num_rounds:
        print(f"\rnum_rounds={num_rounds}", end="")

        anc_meas = []
        data_meas = []
        heralded_cycle = []

        for qubit in all_qubits:
            print(f"\rnum_rounds={num_rounds} qubit={qubit}", end="")

            container = data_manager.get_state_classifier(QubitIDObj(qubit))
            raw_shots = container.shots

            heralded_cycle_idx = idx_kernel.get_heralded_cycle_acquisition_indices(
                qubit_id=QubitIDObj(qubit), cycle_stabilizer_count=num_rounds
            )
            heralded_cycle_data = raw_shots[heralded_cycle_idx]
            heralded_cycle.append([heralded_cycle_data.real, heralded_cycle_data.imag])

            if qubit in metadata["anc_qubits"]:
                anc_meas_idx = idx_kernel.get_stabilizer_acquisition_indices(
                    qubit_id=QubitIDObj(qubit), cycle_stabilizer_count=num_rounds
                )
                anc_meas_data = raw_shots[anc_meas_idx]
                anc_meas.append([anc_meas_data.real, anc_meas_data.imag])

            if qubit in metadata["data_qubits"]:
                data_meas_idx = idx_kernel.get_projected_cycle_acquisition_indices(
                    qubit_id=QubitIDObj(qubit), cycle_stabilizer_count=num_rounds
                )
                data_meas_idx = data_meas_idx[:, 0]  # shape=(num_shots, 1)
                data_meas_data = raw_shots[data_meas_idx]
                data_meas.append([data_meas_data.real, data_meas_data.imag])

        heralded_reps = (
            1 if len(heralded_cycle_data.shape) == 1 else heralded_cycle_data.shape[1]
        )
        num_shots = heralded_cycle_data.shape[0]

        data_init = metadata["data_init"]
        data_init = np.array([data_init[q] for q in metadata["data_qubits"]])
        data_init = data_init.astype(bool)

        ideal_data_meas = deepcopy(data_init)
        if num_rounds % 2 == 1:
            ideal_data_meas = ideal_data_meas ^ 1

        proj_matrix = layout.projection_matrix(stab_type="x_type")
        data_init_xr = xr.DataArray(
            data=data_init,
            coords=dict(data_qubit=metadata["data_qubits"]),
        )
        ideal_anc_meas = (proj_matrix @ data_init_xr) % 2
        # sort them in order
        ideal_anc_meas = ideal_anc_meas.sel(anc_qubit=metadata["anc_qubits"]).values
        ideal_anc_meas = np.repeat(
            ideal_anc_meas[:, np.newaxis], repeats=num_rounds, axis=-1
        )
        # only even rounds should be 0 (starting from qec_round=1)
        ideal_anc_meas[:, 1::2] = 0

        ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("anc_qubit", "iq", "shot", "qec_round"), anc_meas),
                data_meas=(("data_qubit", "iq", "shot"), data_meas),
                heralded_init=(("qubit", "iq", "shot", "heralded_rep"), heralded_cycle),
                ideal_anc_meas=(
                    ("anc_qubit", "qec_round"),
                    ideal_anc_meas.astype(bool),
                ),
                ideal_data_meas=(
                    ("data_qubit",),
                    ideal_data_meas.astype(bool),
                ),
                data_init=(
                    ("data_qubit"),
                    data_init,
                ),
            ),
            coords=dict(
                qubit=all_qubits,
                anc_qubit=metadata["anc_qubits"],
                data_qubit=metadata["data_qubits"],
                shot=list(range(num_shots)),
                qec_round=list(range(1, num_rounds + 1)),
                heralded_rep=list(range(heralded_reps)),
                iq=["I", "Q"],
                rot_basis=metadata["rot_basis"],
                meas_reset=False,
            ),
        )
        ds.to_netcdf(qec_dir / f"iq_data_r{num_rounds}.nc")

    print("\nDone!")
