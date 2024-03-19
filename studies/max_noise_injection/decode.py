print("Importing libraries...")
import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import pymatching
import stim

from qec_util import Layout
from rep_code.defects import get_defect_vector
from rep_code.dataset import sequence_generator


DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
DEM_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/dems"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/output_mwpm"
)

EXP_NAME = "20230119_initial_data_d5"

DEFECTS_NAME = "defects_DecayLinearClassifierFit"
NOISE_NAME = "exp-circ-level_noise"

####################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

with open(DEM_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data_dem = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

(OUTPUT_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / EXP_NAME / "config_data.yaml", "w") as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    dem_dir = DEM_DIR / EXP_NAME / config_data_dem["data"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")

    defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
    defects = defects_xr.defects
    final_defects = defects_xr.final_defects
    log_errors = defects_xr.log_flips.values
    # sort defect data into vector with same ordering
    # as the stim circuit
    defect_vec = get_defect_vector(
        defects,
        final_defects,
        anc_order=layout.get_qubits(role="anc"),
        dim_first="anc_qubit",
    )

    setup = Setup({"name": "", "description": "", "setup": {"sq_error_prob": 0.01}})
    qubit_inds = {q: layout.get_inds([q])[0] for q in layout.get_qubits()}
    model = IncNoiseModelExp(setup, qubit_inds)

    exp_circ = memory_experiment(
        model,
        layout,
        num_rounds=element["num_rounds"],
        data_init={f"D{q}": s for q, s in zip(element["data_qubits"], state)},
        basis=element["basis"],
    )
    sampler = exp_circ.compile_detector_sampler()
    add_defect_vec, add_log_error = sampler.sample(
        shots=len(defects_xr.shot), separate_observables=True
    )

    new_defect_vec = defect_vec ^ add_defect_vec
    new_log_error = log_flips ^ add_log_error

    # dem and mwpm
    dem = stim.DetectorErrorModel.from_file(dem_dir / f"{NOISE_NAME}.dem")
    mwpm = pymatching.Matching(dem)

    # decode
    prediction = mwpm.decode_batch(new_defect_vec)
    # Note: if not flattened, the log_flips is incorrect
    log_flips = new_log_error != prediction.flatten()

    # store logical errors
    log_flips = xr.DataArray(data=log_flips, coords=dict(shot=defects_xr.shot.values))

    output_dir.mkdir(parents=True, exist_ok=True)
    log_flips.to_netcdf(output_dir / f"log_errors_{NOISE_NAME}.nc")

    print(
        f"\033[F\033[K{data_dir} p_L={np.average(log_flips):0.3f}",
        flush=True,
    )
