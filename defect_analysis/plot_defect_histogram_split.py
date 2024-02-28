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
from rep_code.defects.analysis import (
    fit_binomial,
    binomial_dist,
    gaussian_dist,
    fit_gaussian,
    poisson_dist,
    fit_poisson,
)

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/defect_analysis"
)

EXP_NAME = "20230119_initial_data_d5"

DEFECTS_NAME = "defects_DecayLinearClassifierFit"

DISTRIBUTION = "poisson"
COMBINE_ROUNDS = 12
NUM_ROUNDS = 60

############################

print("Running script...")

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]
STATES = STRING_DATA.pop("state")
_ = STRING_DATA.pop("num_rounds")

# remove state from directory names
new_config_data = deepcopy(config_data)
new_config_data["string_data_options"] = STRING_DATA
new_config_data["data"] = (
    new_config_data["data"].replace("_s{state}", "").replace("_r{num_rounds}", "")
)
new_config_data["config"] = (
    new_config_data["config"].replace("_s{state}", "").replace("_r{num_rounds}", "")
)
# simulation datasets do not have readout calibration
if "readout_calibration" in new_config_data:
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

    for state in STATES:
        data_dir = (
            DATA_DIR
            / EXP_NAME
            / config_data["data"].format(**element, state=state, num_rounds=NUM_ROUNDS)
        )
        print(f"\033[F\033[K{data_dir}", flush=True)

        defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
        defects = defects_xr.defects

        combined = {pack: [] for pack in range(NUM_ROUNDS // COMBINE_ROUNDS)}
        for pack in range(NUM_ROUNDS // COMBINE_ROUNDS):
            defect_counts = defects.sel(
                qec_round=np.arange(COMBINE_ROUNDS * pack, COMBINE_ROUNDS * (pack + 1))
                + 1
            ).sum(dim="qec_round")
            combined[pack].append(defect_counts)

    defect_statistics = {}
    for pack in range(NUM_ROUNDS // COMBINE_ROUNDS):
        defect_statistics[pack] = xr.concat(combined[pack], dim="shot")

    # create list of colors
    colormap = plt.cm.rainbow(np.linspace(0, 1, len(defect_statistics)))
    anc_qubits = defects.anc_qubit.values

    for q in anc_qubits:
        fig, ax = plt.subplots()

        for k, pack in enumerate(range(NUM_ROUNDS // COMBINE_ROUNDS)):
            # histogram is difficult when values are integers
            bins, counts = np.unique(
                defect_statistics[pack].sel(anc_qubit=q).values,
                return_counts=True,
            )
            num_rounds = COMBINE_ROUNDS
            # add counts that are 0
            all_bins = np.arange(0, (num_rounds + 1) + 1, dtype=int)
            all_counts = np.zeros_like(all_bins)
            all_counts[bins] = counts
            all_probs = all_counts / np.sum(all_counts)

            x_theo, probs_theo = np.array([]), np.array([])
            if DISTRIBUTION == "binomial":
                p = fit_binomial(all_bins, all_probs, num_rounds + 1)
                x_theo = np.linspace(0, num_rounds + 1, 1_000)
                probs_theo = binomial_dist(x_theo, n=num_rounds + 1, p=p)
            if DISTRIBUTION == "gaussian":
                mu, sigma = fit_gaussian(all_bins, all_probs)
                x_theo = np.linspace(0, num_rounds + 1, 1_000)
                probs_theo = gaussian_dist(x_theo, mu, sigma)
                print(
                    f"R={num_rounds} q={q} mu={mu:0.5f}, sigma={sigma:0.5f}, q={sigma/mu:0.5f}"
                )
            if DISTRIBUTION == "poisson":
                p = fit_poisson(all_bins, all_probs)[0]
                x_theo = np.arange(0, num_rounds + 1).astype(int)
                probs_theo = poisson_dist(x_theo, p)
                print(f"R={num_rounds} p={p:0.5f} p/R={p/num_rounds:0.5f}")

            ax.plot(
                all_bins,
                all_probs,
                label=f"R={pack*COMBINE_ROUNDS}-{(pack+1)*COMBINE_ROUNDS}",
                color=colormap[k],
                linestyle="-",
                marker=".",
            )

            ax.plot(
                x_theo,
                probs_theo,
                color="gray",
                linestyle="--",
                marker="none",
            )

        ax.legend(loc="best")
        ax.set_xlabel("# triggered defects in a sample")
        ax.set_ylabel("probability")
        ax.set_xlim(0, num_rounds + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(ymin=0)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{DEFECTS_NAME}_{q}_histogram_combined_R{NUM_ROUNDS}.pdf",
            format="pdf",
        )

        plt.close()

    for q in anc_qubits:
        fig, ax = plt.subplots()

        for k, pack in enumerate(range(NUM_ROUNDS // COMBINE_ROUNDS)):
            # histogram is difficult when values are integers
            bins, counts = np.unique(
                defect_statistics[pack].sel(anc_qubit=q).values,
                return_counts=True,
            )
            num_rounds = COMBINE_ROUNDS
            # add counts that are 0
            all_bins = np.arange(0, (num_rounds + 1) + 1, dtype=int)
            all_counts = np.zeros_like(all_bins)
            all_counts[bins] = counts
            all_probs = all_counts / np.sum(all_counts)

            x_theo, probs_theo = np.array([]), np.array([])
            if DISTRIBUTION == "binomial":
                p = fit_binomial(all_bins, all_probs, num_rounds + 1)
                x_theo = np.linspace(0, num_rounds + 1, 1_000)
                probs_theo = binomial_dist(x_theo, n=num_rounds + 1, p=p)
            if DISTRIBUTION == "gaussian":
                mu, sigma = fit_gaussian(all_bins, all_probs)
                x_theo = np.linspace(0, num_rounds + 1, 1_000)
                probs_theo = gaussian_dist(x_theo, mu, sigma)
                print(
                    f"R={num_rounds} q={q} mu={mu:0.5f}, sigma={sigma:0.5f}, q={sigma/mu:0.5f}"
                )
            if DISTRIBUTION == "poisson":
                p = fit_poisson(all_bins, all_probs)[0]
                x_theo = np.arange(0, num_rounds + 1).astype(int)
                probs_theo = poisson_dist(x_theo, p)
                print(f"R={num_rounds} p={p:0.5f} p/R={p/num_rounds:0.5f}")

            ax.plot(
                all_bins / (num_rounds + 1),
                all_probs,
                label=f"R={pack*COMBINE_ROUNDS}-{(pack+1)*COMBINE_ROUNDS}",
                color=colormap[k],
                linestyle="-",
                marker=".",
            )

            ax.plot(
                x_theo / (num_rounds + 1),
                probs_theo,
                color="gray",
                linestyle="--",
                marker="none",
            )

        ax.legend(loc="best")
        ax.set_xlabel("# triggered defects in a sample / number of possible defects")
        ax.set_ylabel("probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin=0)

        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"{DEFECTS_NAME}_{q}_histogram_combined_R{NUM_ROUNDS}_relative.pdf",
            format="pdf",
        )

        plt.close()
