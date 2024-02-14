print("Importing libraries...")
import pathlib
import yaml
from copy import deepcopy

import xarray as xr
import matplotlib.pyplot as plt

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

    defects_combined = []
    final_defects_combined = []

    for state in STATES:
        data_dir = (
            DATA_DIR / EXP_NAME / config_data["data"].format(**element, state=state)
        )

        defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
        defects = defects_xr.defects
        final_defects = defects_xr.final_defects

        kargs_plot = {q: dict(color=colors[q]) for q in defects_xr.anc_qubit.values}

        fig, ax = plt.subplots()

        plot_average_defect_rate(ax, defects, final_defects)
        plot_defect_rates(ax, defects, final_defects, **kargs_plot)

        fig.tight_layout()
        fig.savefig(output_dir / f"{DEFECTS_NAME}_rates_s{state}.pdf", format="pdf")

        plt.close()

        defects_combined.append(defects)
        final_defects_combined.append(final_defects)

        print(f"\033[F\033[K{data_dir}", flush=True)

    # plot combined defect rates
    defects_combined = xr.concat(defects_combined, dim="shot")
    final_defects_combined = xr.concat(final_defects_combined, dim="shot")

    kargs_plot = {q: dict(color=colors[q]) for q in defects_xr.anc_qubit.values}

    fig, ax = plt.subplots()

    plot_average_defect_rate(ax, defects_combined, final_defects_combined)
    plot_defect_rates(ax, defects_combined, final_defects_combined, **kargs_plot)

    fig.tight_layout()
    fig.savefig(output_dir / f"{DEFECTS_NAME}_rates_combined.pdf", format="pdf")

    plt.close()
