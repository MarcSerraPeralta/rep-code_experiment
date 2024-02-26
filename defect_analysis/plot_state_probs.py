print("Importing libraries...")
import pathlib
import yaml
from copy import deepcopy

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from qec_util import Layout
from rep_code.dataset import sequence_generator
from rep_code.defects.plots import plot_average_defect_rate, plot_defect_rates

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/defect_analysis"
)

EXP_NAME = "20230119_initial_data_d3"

PROBS_NAME = "state_probs_GaussMixClassifier"

############################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
NUM_ROUNDS = STRING_DATA.pop("num_rounds")

# remove state from directory names
new_config_data = deepcopy(config_data)
new_config_data["string_data_options"] = STRING_DATA
new_config_data["data"] = new_config_data["data"].replace("_r{num_rounds}", "")
new_config_data["config"] = new_config_data["config"].replace("_r{num_rounds}", "")
# simulation datasets do not have readout calibration
if "readout_calibration" in new_config_data:
    new_config_data["readout_calibration"] = new_config_data[
        "readout_calibration"
    ].replace("_r{num_rounds}", "")

(OUTPUT_DIR / EXP_NAME).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / EXP_NAME / "config_data.yaml", "w") as file:
    yaml.dump(new_config_data, file, default_flow_style=False)

colors = {
    "Z1": "#1f77b4",
    "Z2": "#ff7f0e",
    "Z3": "#2ca02c",
    "Z4": "#d62728",
    "X1": "#9467bd",
    "X2": "#8c564b",
    "X3": "#e377c2",
    "X4": "#7f7f7f",
}

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    output_dir = OUTPUT_DIR / EXP_NAME / new_config_data["data"].format(**element)
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load classifier and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    probs_anc_combined = {q: np.zeros(max(NUM_ROUNDS)) for q in anc_qubits}
    probs_data_combined = {q: np.zeros(len(NUM_ROUNDS)) for q in data_qubits}

    counts = np.zeros(len(NUM_ROUNDS))

    for num_rounds in NUM_ROUNDS:
        data_dir = (
            DATA_DIR
            / EXP_NAME
            / config_data["data"].format(**element, num_rounds=num_rounds)
        )
        print(f"\033[F\033[K{data_dir}", flush=True)

        probs_xr = xr.load_dataset(data_dir / f"{PROBS_NAME}.nc")

        for anc_qubit in anc_qubits:
            probs_anc_combined[anc_qubit][:num_rounds] += probs_xr.probs_anc.sel(
                anc_qubit=anc_qubit, state=2
            )
        counts[:num_rounds] += 1

        for data_qubit in data_qubits:
            probs_data_combined[data_qubit][
                NUM_ROUNDS.index(num_rounds)
            ] = probs_xr.probs_data.sel(data_qubit=data_qubit, state=2)

    for anc_qubit in anc_qubits:
        probs_anc_combined[anc_qubit] /= counts

    # plot
    fig, ax = plt.subplots()

    for anc_qubit in anc_qubits:
        ax.plot(
            np.arange(1, max(NUM_ROUNDS) + 1),
            probs_anc_combined[anc_qubit],
            label=f"{anc_qubit}",
            color=colors[anc_qubit],
            linestyle="-",
            marker=".",
        )

    for data_qubit in data_qubits:
        ax.plot(
            NUM_ROUNDS,
            probs_data_combined[data_qubit],
            label=f"{data_qubit}",
            # color=colors[anc_qubit],
            linestyle="--",
            marker=".",
        )

    ax.legend(loc="best")
    ax.set_xlabel("QEC round (R for data qubits, r for ancilla qubits)")
    ax.set_ylabel("probability of state 2")
    ax.set_xlim(0, max(NUM_ROUNDS) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ymin=0)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{PROBS_NAME}_prob_state_2.pdf",
        format="pdf",
    )

    plt.close()
