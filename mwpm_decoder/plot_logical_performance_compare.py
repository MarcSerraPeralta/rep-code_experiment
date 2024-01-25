import pathlib
import os

import pathlib
import yaml

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from rep_code.decoding import (
    plot_fidelity_exp,
    plot_fidelity_fit,
    get_error_rate,
    plot_error_prob_fit,
    plot_error_prob_exp,
)
from rep_code.dataset import sequence_generator


CONFIG_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/output_mwpm"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

LOG_ERR_NAME_1 = "log_errors_estimated_noise_TwoStateLinearClassifierFit"
LOG_ERR_NAME_2 = "log_errors_estimated_noise_DecayLinearClassifierFit"

####################

with open(CONFIG_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
NUM_ROUNDS = STRING_DATA.pop("num_rounds")

list_log_errors_1 = {r: [] for r in NUM_ROUNDS}
list_log_errors_2 = {r: [] for r in NUM_ROUNDS}

print("\n", end="")  # for printing purposes

for num_rounds in sequence_generator({"num_rounds": NUM_ROUNDS}):
    for element in sequence_generator(STRING_DATA):
        data_dir = (
            DATA_DIR / EXP_NAME / config_data["data"].format(**num_rounds, **element)
        )

        # load logical errors
        log_errors = xr.load_dataarray(data_dir / f"{LOG_ERR_NAME_1}.nc")
        list_log_errors_1[num_rounds["num_rounds"]] += log_errors.values.tolist()

        log_errors = xr.load_dataarray(data_dir / f"{LOG_ERR_NAME_2}.nc")
        list_log_errors_2[num_rounds["num_rounds"]] += log_errors.values.tolist()

        print(
            f"\033[F\033[K{config_data['data'].format(**element, **num_rounds)} p_L={np.average(log_errors):0.3f}",
            flush=True,
        )

list_log_errors_1 = [
    np.array(list_log_errors_1[r]).astype(bool) ^ 1 for r in NUM_ROUNDS
]
distance = STRING_DATA["distance"]

error_rate, r0 = get_error_rate(NUM_ROUNDS, list_log_errors_1, distance, return_r0=True)

fig, ax = plt.subplots()

plot_fidelity_exp(
    ax,
    NUM_ROUNDS,
    list_log_errors_1,
    color="red",
    linestyle="",
    fmt=".",
    label=LOG_ERR_NAME_1,
)
plot_fidelity_fit(
    ax,
    NUM_ROUNDS,
    error_rate.nominal_value,
    r0.nominal_value,
    distance,
    color="red",
    label=f"err_rate = {error_rate*100}%",
    linestyle="--",
)

list_log_errors_2 = [
    np.array(list_log_errors_2[r]).astype(bool) ^ 1 for r in NUM_ROUNDS
]
distance = STRING_DATA["distance"]

error_rate, r0 = get_error_rate(NUM_ROUNDS, list_log_errors_2, distance, return_r0=True)

plot_fidelity_exp(
    ax,
    NUM_ROUNDS,
    list_log_errors_2,
    color="blue",
    linestyle="",
    fmt=".",
    label=LOG_ERR_NAME_2,
)
plot_fidelity_fit(
    ax,
    NUM_ROUNDS,
    error_rate.nominal_value,
    r0.nominal_value,
    distance,
    color="blue",
    label=f"err_rate = {error_rate*100}%",
    linestyle="--",
)

ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(
    DATA_DIR / EXP_NAME / f"{LOG_ERR_NAME_1}_vs_{LOG_ERR_NAME_2}.pdf", format="pdf"
)

plt.show()
