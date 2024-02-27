print("Importing libraries...")
import pathlib
import yaml
from copy import deepcopy

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from qec_util import Layout
from iq_readout.two_state_classifiers import *
from rep_code.dataset import sequence_generator
from rep_code.nn_decoder.processing import to_defect_probs_leakage_IQ

DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/defect_analysis"
)

EXP_NAME = "20230119_initial_data_d5"

IQ_DATA_NAME = "iq_data"
CLASSIFIER = GaussMixLinearClassifier


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
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    cal_dir = DATA_DIR / EXP_NAME / config_data["readout_calibration"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load classifier and layout
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    proj_mat = layout.projection_matrix(stab_type="x_type")
    anc_qubits = layout.get_qubits(role="anc")

    cla_name = CLASSIFIER.__name__
    cla_params = np.load(
        cal_dir / f"2state_{cla_name}_params_ps.npy", allow_pickle=True
    ).item()
    classifiers = {q: CLASSIFIER().load(cla_params[q]) for q in layout.get_qubits()}

    # get defect probabilities
    BINS = np.linspace(0, 1, 100)
    prob_defects = {q: np.zeros_like(BINS[:-1]) for q in anc_qubits}

    for num_rounds in NUM_ROUNDS:
        d_probs_comb = []

        for state in STATES:
            data_dir = (
                DATA_DIR
                / EXP_NAME
                / config_data["data"].format(
                    **element, state=state, num_rounds=num_rounds
                )
            )
            print(f"\033[F\033[K{data_dir}", flush=True)

            iq_data_xr = xr.load_dataset(data_dir / f"{IQ_DATA_NAME}.nc")
            defect_probs, _, _ = to_defect_probs_leakage_IQ(
                iq_data_xr,
                proj_mat=proj_mat,
                two_state_classifiers=classifiers,
                three_state_classifiers=None,
                leakage={"data": False, "anc": False},
                digitization={"data": True, "anc": False},
            )
            d_probs_comb.append(defect_probs)

        # combine data from all states
        d_probs_comb = xr.concat(d_probs_comb, dim="shot")
        d_probs_comb = d_probs_comb.stack(item=["shot", "qec_round"])

        for anc_qubit in anc_qubits:
            values = d_probs_comb.sel(anc_qubit=anc_qubit).values
            counts, _ = np.histogram(values, bins=BINS)
            prob_defects[anc_qubit] += counts

    # plot the results
    fig, ax = plt.subplots()
    bin_centers = 0.5 * (BINS[:-1] + BINS[1:])

    for anc_qubit in anc_qubits:
        ax.plot(
            bin_centers,
            prob_defects[anc_qubit],
            label=f"{anc_qubit}",
            color=colors[anc_qubit],
            linestyle="-",
            marker=".",
        )

    prob_defects = np.sum([prob_defects[q] for q in anc_qubits], axis=0)
    ax.plot(
        bin_centers,
        prob_defects,
        label=f"all",
        color="black",
        linestyle="-",
        marker=".",
    )

    ax.legend(loc="best")
    ax.set_xlabel("defect probability")
    ax.set_ylabel("# counts")
    ax.set_xlim(0, 1)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"histogram_defect_prob_{cla_name}.pdf",
        format="pdf",
    )

    plt.close()
