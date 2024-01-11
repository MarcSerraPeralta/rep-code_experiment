print("Importing libraries...")
import pathlib
from copy import deepcopy

import numpy as np
import xarray as xr

from qce_interp import DataManager, QubitIDObj, ParityType, Surface17Layer, StateKey
from rep_code.layout import get_rep_code_layout

ROUNDS = list(range(1, 60 + 1))
READOUT_CALIBRATION_STATES = [StateKey.STATE_0, StateKey.STATE_1, StateKey.STATE_2]
DATA_QUBITS = [QubitIDObj("D7"), QubitIDObj("D4"), QubitIDObj("D5")]
ANC_QUBITS = [QubitIDObj("Z3"), QubitIDObj("Z1")]
ROT_BASIS = False
DATA_INIT = [0, 1, 0]  # same order as DATA_QUBITS
RUN_DIR = "014754_Repeated_stab_meas"

# get path
if len(DATA_INIT) == 3:
    data_init_ = "".join(map(str, DATA_INIT + DATA_INIT[1:3]))
    INPUT_DIR = f"distance3 {data_init_}"
elif len(DATA_INIT) == 5:
    data_init_ = "".join(map(str, DATA_INIT))
    INPUT_DIR = f"distance5 {data_init_}"

INPUT_DIR = pathlib.Path(INPUT_DIR)
FILE_PATH = INPUT_DIR / RUN_DIR / f"{RUN_DIR}.hdf5"

OUTPUT_DIR = INPUT_DIR / f"{RUN_DIR}_iq"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# add layout file
QUBITS = [q.name for q in DATA_QUBITS + ANC_QUBITS]
layout = get_rep_code_layout(QUBITS)
layout.to_yaml(OUTPUT_DIR / "rep_code_layout.yaml")

# load data
print("Loading DataManager...")
data_manager = DataManager.from_file_path(
    file_path=FILE_PATH,
    rounds=ROUNDS,
    heralded_initialization=True,
    qutrit_calibration_points=True,
    involved_data_qubit_ids=DATA_QUBITS,
    involved_ancilla_qubit_ids=ANC_QUBITS,
    expected_parity_lookup={q: ParityType.ODD for q in DATA_QUBITS},
    device_layout=Surface17Layer(),
)

# extract data
qubit_ids = data_manager.involved_qubit_ids
idx_kernel = data_manager._experiment_index_kernel

# calibration data
print("Converting calibration data to xarray...")

readout_calibration = []
heralded_calibration = []

for qubit in qubit_ids:
    print(f"\rqubit={qubit}", end="")
    container = data_manager.get_state_classifier(qubit)
    raw_shots = container.shots
    readout_calibration_qubit = []
    heralded_calibration_qubit = []
    for state in READOUT_CALIBRATION_STATES:
        print(f"\rqubit={qubit} state={state}", end="")
        readout_calibration_idx = (
            idx_kernel.get_projected_calibration_acquisition_indices(
                qubit_id=qubit, state=state
            )
        )
        readout_calibration_data = raw_shots[readout_calibration_idx]
        readout_calibration_qubit.append(
            [readout_calibration_data.real, readout_calibration_data.imag]
        )
        heralded_calibration_idx = (
            idx_kernel.get_heralded_calibration_acquisition_indices(
                qubit_id=qubit, state=state
            )
        )
        heralded_calibration_data = raw_shots[heralded_calibration_idx]
        heralded_calibration_qubit.append(
            [heralded_calibration_data.real, heralded_calibration_data.imag]
        )
    readout_calibration.append(readout_calibration_qubit)
    heralded_calibration.append(heralded_calibration_qubit)

heralded_reps = (
    1
    if len(heralded_calibration_data.shape) == 1
    else heralded_calibration_data.shape[1]
)

ds = xr.Dataset(
    data_vars=dict(
        calibration=(("qubit", "state", "iq", "shot"), readout_calibration),
        heralded_init=(("qubit", "state", "iq", "shot"), heralded_calibration),
    ),
    coords=dict(
        state=[i.value for i in READOUT_CALIBRATION_STATES],
        qubit=[q.name for q in qubit_ids],
        shot=list(range(heralded_calibration_data.shape[0])),
        heralded_rep=list(range(heralded_reps)),
        iq=["I", "Q"],
    ),
)
ds.to_netcdf(OUTPUT_DIR / "calibration.nc")


# QEC cycle data
print("\nConverting QEC data to xarray...")

for num_rounds in ROUNDS:
    print(f"\rnum_rounds={num_rounds}", end="")

    anc_meas = []
    data_meas = []
    heralded_cycle = []

    for qubit in qubit_ids:
        print(f"\rnum_rounds={num_rounds} qubit={qubit}", end="")

        container = data_manager.get_state_classifier(qubit)
        raw_shots = container.shots

        heralded_cycle_idx = idx_kernel.get_heralded_cycle_acquisition_indices(
            qubit_id=qubit, cycle_stabilizer_count=num_rounds
        )
        heralded_cycle_data = raw_shots[heralded_cycle_idx]
        heralded_cycle.append([heralded_cycle_data.real, heralded_cycle_data.imag])

        if qubit in ANC_QUBITS:
            anc_meas_idx = idx_kernel.get_stabilizer_acquisition_indices(
                qubit_id=qubit, cycle_stabilizer_count=num_rounds
            )
            anc_meas_data = raw_shots[anc_meas_idx]
            anc_meas.append([anc_meas_data.real, anc_meas_data.imag])

        if qubit in DATA_QUBITS:
            data_meas_idx = idx_kernel.get_projected_cycle_acquisition_indices(
                qubit_id=qubit, cycle_stabilizer_count=num_rounds
            )
            data_meas_idx = data_meas_idx[:, 0]  # shape=(num_shots, 1)
            data_meas_data = raw_shots[data_meas_idx]
            data_meas.append([data_meas_data.real, data_meas_data.imag])

    heralded_reps = (
        1 if len(heralded_cycle_data.shape) == 1 else heralded_cycle_data.shape[1]
    )
    num_shots = heralded_cycle_data.shape[0]

    data_init = np.array(DATA_INIT).astype(bool)

    ideal_data_meas = deepcopy(data_init)
    if num_rounds % 2 == 1:
        ideal_data_meas = ideal_data_meas ^ 1

    proj_matrix = layout.projection_matrix(stab_type="x_type")
    data_init_xr = xr.DataArray(
        data=data_init,
        coords=dict(data_qubit=[q.name for q in DATA_QUBITS]),
    )
    ideal_anc_meas = proj_matrix @ data_init_xr
    # sort them in order
    ideal_anc_meas = ideal_anc_meas.sel(anc_qubit=[q.name for q in ANC_QUBITS]).values
    ideal_anc_meas = np.repeat(
        ideal_anc_meas[:, np.newaxis], repeats=num_rounds, axis=-1
    )

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
            qubit=[q.name for q in qubit_ids],
            anc_qubit=[q.name for q in ANC_QUBITS],
            data_qubit=[q.name for q in DATA_QUBITS],
            shot=list(range(num_shots)),
            qec_round=list(range(1, num_rounds + 1)),
            heralded_rep=list(range(heralded_reps)),
            iq=["I", "Q"],
            rot_basis=ROT_BASIS,
        ),
    )
    ds.to_netcdf(OUTPUT_DIR / f"iq_data_r{num_rounds}.nc")

print("\nDone!")
