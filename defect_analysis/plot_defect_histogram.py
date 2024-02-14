print("Importing libraries...")
import pathlib
import yaml

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
    defects = defects_xr.defects
    final_defects = defects_xr.final_defects

    # compute statistics
    defect_counts = defects.sum(dim="qec_round") + final_defects
    defect_counts = defect_counts.transpose("anc_qubit", "shot")
    anc_qubits = defects.anc_qubit.values
    num_rounds = defects.qec_round.values
    # add final round
    num_rounds = np.concatenate([num_rounds, [np.max(num_rounds) + 1]])

    fig, ax = plt.subplots()

    for k, q in enumerate(anc_qubits):
        bins, counts = np.unique(
            defect_counts.sel(anc_qubit=q).values, return_counts=True
        )
        # print(hist, bin_edges, num_rounds)
        ax.plot(bins, counts, colors[k], label=q, linestyle="-", marker=".")

    ax.legend(loc="best")
    ax.set_xlabel("number of triggered defects in a run")
    ax.set_ylabel("number of runs")
    ax.set_xlim(0, np.max(num_rounds) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ymin=0)

    fig.tight_layout()
    fig.savefig(output_dir / f"{DEFECTS_NAME}_histogram.pdf", format="pdf")

    plt.close()

    print(f"\033[F\033[K{data_dir}", flush=True)