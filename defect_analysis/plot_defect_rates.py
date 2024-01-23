import pathlib
import yaml

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

EXP_NAME = "20230119_initial_data_d3_s010_combined"

DEFECTS_NAME = "defects_DecayLinearClassifierFit"

############################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    defects_xr = xr.load_dataset(data_dir / f"{DEFECTS_NAME}.nc")
    defects = defects_xr.defects
    final_defects = defects_xr.final_defects

    fig, ax = plt.subplots()

    plot_average_defect_rate(ax, defects, final_defects)
    plot_defect_rates(ax, defects, final_defects)

    fig.tight_layout()
    fig.savefig(output_dir / f"{DEFECTS_NAME}_rates.pdf", format="pdf")

    plt.close()

    print(
        f"\033[F\033[K{config_data['data'].format(**element)}",
        flush=True,
    )
