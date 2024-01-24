print("Importing libraries...")
import pathlib
import yaml

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qec_util import Layout
from rep_code.dataset import sequence_generator
from rep_code.decoding import get_error_rate


DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/output_mwpm"
)

EXP_NAME = "20230123_error_budget_simulation_d5"

CONFIGS = {
    "config_data_scan_sq_error_prob.yaml": "sq_error_prob",
    "config_data_scan_cz_error_prob.yaml": "cz_error_prob",
}
NOISE_MODEL = "ExperimentalNoiseModelExp"
PLOT_NAME = f"plot_epsL_vs_noise_{NOISE_MODEL}"

####################

print("Running script...")

print("\n", end="")  # for printing purposes

all_data = {}

for config_data_name, noise_param in CONFIGS.items():
    with open(DATA_DIR / EXP_NAME / config_data_name, "r") as file:
        config_data = yaml.safe_load(file)

    STRING_DATA = config_data["string_data_options"]

    list_num_rounds = STRING_DATA.pop("num_rounds")
    list_noise_param = STRING_DATA[noise_param]
    log_err_rate_param = {}

    for element in sequence_generator(STRING_DATA):
        list_log_errors = {r: [] for r in list_num_rounds}

        for num_rounds in list_num_rounds:
            data_dir = (
                DATA_DIR
                / EXP_NAME
                / config_data["data"].format(**element, num_rounds=num_rounds)
            )

            log_errors = xr.load_dataarray(data_dir / f"log_err_{NOISE_MODEL}.nc")
            list_log_errors[num_rounds] = log_errors.values.tolist()

            print(
                f"\033[F\033[K{config_data['data'].format(**element, num_rounds=num_rounds)}",
                flush=True,
            )

        list_log_errors = [
            np.array(list_log_errors[r]).astype(bool) for r in list_num_rounds
        ]
        distance = STRING_DATA["distance"]
        error_rate = get_error_rate(list_num_rounds, list_log_errors, distance)
        log_err_rate_param[element[noise_param]] = error_rate

    all_data[noise_param] = log_err_rate_param

# print the data
fig, ax = plt.subplots()

for noise_param, data in all_data.items():
    param_values = sorted(data.keys())
    eps = [data[v].nominal_value for v in param_values]
    eps_err = [data[v].std_dev for v in param_values]
    delta_p = np.array(param_values) - param_values[0]
    ax.errorbar(delta_p, eps, yerr=eps_err, label=noise_param)

ax.set_xlabel("delta p noise")
ax.set_ylabel("logical error rate")
ax.set_xlim(xmin=0)
ax.legend(loc="best")
fig.tight_layout()
plt.show()
