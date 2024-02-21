print("Importing libraries...")
import pathlib
import yaml
from copy import deepcopy

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from rep_code.dataset import sequence_generator
from rep_code.defects.plots import plot_average_defect_rate, plot_defect_rates

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/defect_analysis"
)

EXP_NAME = "20230119_initial_data_d3"

DEFECTS_NAME = "defects_DecayLinearClassifierFit"

############################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
STATES = STRING_DATA.pop("state")

# remove state from directory names
new_config_data = deepcopy(config_data)
new_config_data["string_data_options"] = STRING_DATA
new_config_data["data"] = new_config_data["data"].replace("_s{state}", "")
new_config_data["config"] = new_config_data["config"].replace("_s{state}", "")
new_config_data["readout_calibration"] = new_config_data["readout_calibration"].replace(
    "_s{state}", ""
)

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
    output_dir.mkdir(parents=True, exist_ok=True)

    defect_counts_combined = []

    for state in STATES:
        data_dir = (
            DATA_DIR / EXP_NAME / config_data["data"].format(**element, state=state)
        )
        print(f"\033[F\033[K{data_dir}", flush=True)

        defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
        defects = defects_xr.defects
        final_defects = defects_xr.final_defects

        # compute statistics
        defect_counts = defects.sum(dim="qec_round") + final_defects
        anc_qubits = defects.anc_qubit.values
        num_rounds = defects.qec_round.values
        # add final round
        num_rounds = np.concatenate([num_rounds, [np.max(num_rounds) + 1]])

        fig, ax = plt.subplots()

        for q in anc_qubits:
            # histogram is difficult when values are integers
            bins, counts = np.unique(
                defect_counts.sel(anc_qubit=q).values, return_counts=True
            )
            # add counts that are 0
            all_bins = np.arange(0, np.max(num_rounds) + 1, dtype=int)
            all_counts = np.zeros_like(all_bins)
            all_counts[bins] = counts
            ax.plot(all_bins, all_counts, colors[q], label=q, linestyle="-", marker=".")

        ax.legend(loc="best")
        ax.set_xlabel("number of triggered defects in a run")
        ax.set_ylabel("number of runs")
        ax.set_xlim(0, np.max(num_rounds) + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(ymin=0)

        fig.tight_layout()
        fig.savefig(output_dir / f"{DEFECTS_NAME}_histogram_s{state}.pdf", format="pdf")

        plt.close()

        defect_counts_combined.append(defect_counts)

    # combine data from all states
    defect_counts_combined = xr.concat(defect_counts_combined, dim="shot")

    fig, ax = plt.subplots()

    for q in anc_qubits:
        # histogram is difficult when values are integers
        bins, counts = np.unique(
            defect_counts_combined.sel(anc_qubit=q).values, return_counts=True
        )
        # add counts that are 0
        all_bins = np.arange(0, np.max(num_rounds) + 1, dtype=int)
        all_counts = np.zeros_like(all_bins)
        all_counts[bins] = counts
        ax.plot(all_bins, all_counts, colors[q], label=q, linestyle="-", marker=".")

    ax.legend(loc="best")
    ax.set_xlabel("# triggered defects in a sample")
    ax.set_ylabel("# samples")
    ax.set_xlim(0, np.max(num_rounds) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ymin=0)

    fig.tight_layout()
    fig.savefig(output_dir / f"{DEFECTS_NAME}_histogram_combined.pdf", format="pdf")

    plt.close()
