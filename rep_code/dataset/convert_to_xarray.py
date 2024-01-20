from typing import List, Dict

import pathlib
from copy import deepcopy

import numpy as np
import xarray as xr

from qce_interp import DataManager, QubitIDObj, StateKey


def calibration_to_xarray(
    data_manager: DataManager, calibration_states: List[int], qubits: List[int]
) -> xr.Dataset:
    """
    Returns the readout calibration data in the DataManager class
    structured in an xarray.Dataset
    """
    idx_kernel = data_manager._experiment_index_kernel

    meas_cal, heral_cal = [], []

    for qubit in qubits:
        print(f"\rqubit={qubit}", end="")
        raw_shots = data_manager.get_state_classifier(QubitIDObj(qubit)).shots
        meas_cal_qubit, heral_cal_qubit = [], []
        for state in calibration_states:
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
            if len(heral_cal_data.shape) == 1:
                heral_cal_data = heral_cal_data.reshape(len(heral_cal_data), 1)
            heral_cal_qubit.append([heral_cal_data.real, heral_cal_data.imag])
        meas_cal.append(meas_cal_qubit)
        heral_cal.append(heral_cal_qubit)

    heral_shots, heral_reps = heral_cal_data.shape

    calibration_ds = xr.Dataset(
        data_vars=dict(
            calibration=(("qubit", "state", "iq", "shot"), meas_cal),
            heralded_init=(("qubit", "state", "iq", "shot", "heralded_rep"), heral_cal),
        ),
        coords=dict(
            state=calibration_states,
            qubit=qubits,
            shot=list(range(heral_shots)),
            heralded_rep=list(range(heral_reps)),
            iq=["I", "Q"],
        ),
    )

    calibration_ds = calibration_ds.transpose("qubit", "state", "shot", "heralded_rep", "iq")

    return calibration_ds


def qec_to_xarray(
    data_manager: DataManager,
    data_qubits: List[str],
    anc_qubits: List[str],
    num_rounds: List[int],
    data_init: Dict[str, int],
    proj_matrix: xr.DataArray,
) -> xr.Dataset:
    """
    Returns the QEC data in the DataManager class structured in an xarray.Dataset
    """
    all_qubits = data_qubits + anc_qubits
    idx_kernel = data_manager._experiment_index_kernel

    data_init = np.array([data_init[q] for q in data_qubits])
    data_init = xr.DataArray(
        data=data_init,
        coords=dict(data_qubit=data_qubits),
    )

    # only even rounds should be 0, starting from qec_round=1
    # while array starts at 0, thus 1::2
    ideal_anc_meas_vec = (proj_matrix @ data_init) % 2
    ideal_anc_meas_vec = ideal_anc_meas_vec.sel(anc_qubit=anc_qubits).values
    ideal_anc_meas = np.repeat(
        ideal_anc_meas_vec[:, np.newaxis], repeats=num_rounds, axis=-1
    )
    ideal_anc_meas[:, 1::2] = 0

    ideal_data_meas = deepcopy(data_init)
    if num_rounds % 2 == 1:
        ideal_data_meas = ideal_data_meas ^ 1

    # convert to numpy arrays for xr.Dataset creation
    ideal_data_meas = ideal_data_meas.values
    data_init = data_init.values

    anc_meas = []
    data_meas = []
    heralded_cycle = []

    for qubit in all_qubits:
        raw_shots = data_manager.get_state_classifier(QubitIDObj(qubit)).shots

        heral_idx = idx_kernel.get_heralded_cycle_acquisition_indices(
            qubit_id=QubitIDObj(qubit), cycle_stabilizer_count=num_rounds
        )
        heral_data = raw_shots[heral_idx]
        if len(heral_data.shape) == 1:
            heral_data = heral_data.reshape(len(heral_data), 1)
        heralded_cycle.append([heral_data.real, heral_data.imag])

        if qubit in anc_qubits:
            anc_meas_idx = idx_kernel.get_stabilizer_acquisition_indices(
                qubit_id=QubitIDObj(qubit), cycle_stabilizer_count=num_rounds
            )
            anc_meas_data = raw_shots[anc_meas_idx]
            anc_meas.append([anc_meas_data.real, anc_meas_data.imag])

        if qubit in data_qubits:
            data_meas_idx = idx_kernel.get_projected_cycle_acquisition_indices(
                qubit_id=QubitIDObj(qubit), cycle_stabilizer_count=num_rounds
            )
            data_meas_idx = data_meas_idx[:, 0]  # shape=(num_shots, 1)
            data_meas_data = raw_shots[data_meas_idx]
            data_meas.append([data_meas_data.real, data_meas_data.imag])

    num_shots, heral_reps = heral_data.shape

    qec_ds = xr.Dataset(
        data_vars=dict(
            anc_meas=(("anc_qubit", "iq", "shot", "qec_round"), anc_meas),
            data_meas=(("data_qubit", "iq", "shot"), data_meas),
            heralded_init=(("qubit", "iq", "shot", "heralded_rep"), heralded_cycle),
            ideal_anc_meas=(
                ("anc_qubit", "qec_round"),
                ideal_anc_meas.astype(bool),
            ),
            ideal_data_meas=(
                ("data_qubit"),
                ideal_data_meas.astype(bool),
            ),
            data_init=(
                ("data_qubit"),
                data_init.astype(bool),
            ),
        ),
        coords=dict(
            qubit=all_qubits,
            anc_qubit=anc_qubits,
            data_qubit=data_qubits,
            shot=list(range(num_shots)),
            qec_round=list(range(1, num_rounds + 1)),
            heralded_rep=list(range(heral_reps)),
            iq=["I", "Q"],
        ),
    )

    qec_ds = qec_ds.transpose(
        "shot",
        "qec_round",
        "anc_qubit",
        "data_qubit",
        "qubit",
        "heralded_rep",
        "iq",
    )

    return qec_ds
