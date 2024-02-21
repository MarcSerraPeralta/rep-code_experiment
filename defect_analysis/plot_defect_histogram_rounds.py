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
NUM_ROUNDS = STRING_DATA.pop("num_rounds")

# remove state from directory names
new_config_data = deepcopy(config_data)
new_config_data["string_data_options"] = STRING_DATA
new_config_data["data"] = (
    new_config_data["data"].replace("_s{state}", "").replace("_r{num_rounds}", "")
)
new_config_data["config"] = (
    new_config_data["config"].replace("_s{state}", "").replace("_r{num_rounds}", "")
)
new_config_data["readout_calibration"] = (
    new_config_data["readout_calibration"]
    .replace("_s{state}", "")
    .replace("_r{num_rounds}", "")
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

    defect_statistics = {}

    for num_rounds in NUM_ROUNDS:
        defect_counts_combined = []

        for state in STATES:
            data_dir = (
                DATA_DIR
                / EXP_NAME
                / config_data["data"].format(
                    **element, state=state, num_rounds=num_rounds
                )
            )
            print(f"\033[F\033[K{data_dir}", flush=True)

            defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
            defects = defects_xr.defects
            final_defects = defects_xr.final_defects

            # compute statistics
            defect_counts = defects.sum(dim="qec_round") + final_defects
            qec_rounds = defects.qec_round.values
            # add final round
            qec_rounds = np.concatenate([qec_rounds, [np.max(qec_rounds) + 1]])

            defect_counts_combined.append(defect_counts)

        # combine data from all states
        defect_statistics[num_rounds] = xr.concat(defect_counts_combined, dim="shot")

    # create list of colors
    colormap = plt.cm.rainbow(np.linspace(0, 1, len(NUM_ROUNDS)))
    PLOT_MULTIPLES = 5
    anc_qubits = defects.anc_qubit.values

    for q in anc_qubits:
        fig, ax = plt.subplots()

        for k, num_rounds in enumerate(NUM_ROUNDS):
            if num_rounds < 10:
                continue
            if num_rounds % PLOT_MULTIPLES != 0:
                continue

            # histogram is difficult when values are integers
            bins, counts = np.unique(
                defect_statistics[num_rounds].sel(anc_qubit=q).values,
                return_counts=True,
            )
            # add counts that are 0
            all_bins = np.arange(0, (num_rounds + 1) + 1, dtype=int)
            all_counts = np.zeros_like(all_bins)
            all_counts[bins] = counts
            all_counts = all_counts / np.sum(all_counts)
            ax.plot(
                all_bins,
                all_counts,
                label=f"R={num_rounds}" if num_rounds % PLOT_MULTIPLES == 0 else None,
                color=colormap[k],
                linestyle="-",
                marker=".",
            )

        ax.legend(loc="best")
        ax.set_xlabel("# triggered defects in a sample")
        ax.set_ylabel("probability")
        ax.set_xlim(0, num_rounds + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(ymin=0)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{DEFECTS_NAME}_{q}_histogram_combined_per_round.pdf",
            format="pdf",
        )

        plt.close()

    for q in anc_qubits:
        fig, ax = plt.subplots()

        for k, num_rounds in enumerate(NUM_ROUNDS):
            if num_rounds < 10:
                continue
            if num_rounds % PLOT_MULTIPLES != 0:
                continue

            # histogram is difficult when values are integers
            bins, counts = np.unique(
                defect_statistics[num_rounds].sel(anc_qubit=q).values,
                return_counts=True,
            )
            # add counts that are 0
            all_bins = np.arange(0, (num_rounds + 1) + 1, dtype=int)
            all_counts = np.zeros_like(all_bins)
            all_counts[bins] = counts
            all_counts = all_counts / np.sum(all_counts)
            ax.plot(
                all_bins / (num_rounds + 1),
                all_counts,
                label=f"R={num_rounds}" if num_rounds % PLOT_MULTIPLES == 0 else None,
                color=colormap[k],
                linestyle="-",
                marker=".",
            )

        ax.legend(loc="best")
        ax.set_xlabel("# triggered defects in a sample / number of rounds")
        ax.set_ylabel("probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin=0)

        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"{DEFECTS_NAME}_{q}_histogram_combined_per_round_relative.pdf",
            format="pdf",
        )

        plt.close()
