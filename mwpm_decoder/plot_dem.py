import pathlib
import os
import yaml

import numpy as np
import xarray as xr
import pymatching
import stim
import matplotlib.pyplot as plt

from qec_util import Layout
from rep_code.dataset import sequence_generator
from rep_code.decoding import plot_dem, plot_dem_difference


DATA_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/output_mwpm"
)

EXP_NAME = "20230119_initial_data_d3_s010_combined"

NOISE_NAME_1 = "t1t2_noise"
NOISE_NAME_2 = "estimated_noise_DecayLinearClassifierFit"

####################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    config_dir = DATA_DIR / EXP_NAME / config_data["config"].format(**element)
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir = OUTPUT_DIR / EXP_NAME / config_data["data"].format(**element)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load
    layout = Layout.from_yaml(config_dir / "rep_code_layout.yaml")
    figsize = (element["distance"], max(element["num_rounds"] / 2, 2))

    # plot dem 1
    dem1 = stim.DetectorErrorModel.from_file(data_dir / f"{NOISE_NAME_1}.dem")
    fig, ax = plt.subplots(figsize=figsize)
    plot_dem(ax, dem1)
    ax.set_ylim(-1, element["num_rounds"] + 1)
    fig.tight_layout()
    fig.savefig(output_dir / f"dem_{NOISE_NAME_1}.pdf", format="pdf")
    plt.close()

    # plot dem 2
    dem2 = stim.DetectorErrorModel.from_file(data_dir / f"{NOISE_NAME_2}.dem")
    fig, ax = plt.subplots(figsize=figsize)
    plot_dem(ax, dem2)
    ax.set_ylim(-1, element["num_rounds"] + 1)
    fig.tight_layout()
    fig.savefig(output_dir / f"dem_{NOISE_NAME_2}.pdf", format="pdf")
    plt.close()

    # plot comparison
    fig, ax = plt.subplots(figsize=figsize)
    plot_dem_difference(ax, dem1, dem2)
    ax.set_ylim(-1, element["num_rounds"] + 1)
    fig.tight_layout()
    fig.savefig(output_dir / f"dem_{NOISE_NAME_1}_vs_{NOISE_NAME_2}.pdf", format="pdf")
    plt.close()

    print(f"\033[F\033[K{config_data['data'].format(**element)}", flush=True)
