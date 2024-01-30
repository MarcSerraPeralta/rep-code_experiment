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
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/output_mwpm"
)

EXP_NAME = "20230119_initial_data_d3"

NOISE_NAME_1 = "exp-circ-level_noise"
NOISE_NAME_2 = "estimated_noise_DecayLinearClassifierFit"

####################

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)

STRING_DATA = config_data["string_data_options"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n", end="")  # for printing purposes

for element in sequence_generator(STRING_DATA):
    data_dir = DATA_DIR / EXP_NAME / config_data["data"].format(**element)
    print(f"\033[F\033[K{data_dir}", flush=True)

    # figure inputs
    figsize = (element["distance"], max(element["num_rounds"], 2))

    # plot dem 1
    dem1 = stim.DetectorErrorModel.from_file(data_dir / f"{NOISE_NAME_1}.dem")
    fig, ax = plt.subplots(figsize=figsize)
    plot_dem(ax, dem1)
    ax.set_ylim(-1, element["num_rounds"] + 1)
    fig.tight_layout()
    fig.savefig(data_dir / f"dem_{NOISE_NAME_1}.pdf", format="pdf")
    plt.close()

    # plot dem 2
    dem2 = stim.DetectorErrorModel.from_file(data_dir / f"{NOISE_NAME_2}.dem")
    fig, ax = plt.subplots(figsize=figsize)
    plot_dem(ax, dem2)
    ax.set_ylim(-1, element["num_rounds"] + 1)
    fig.tight_layout()
    fig.savefig(data_dir / f"dem_{NOISE_NAME_2}.pdf", format="pdf")
    plt.close()

    # plot comparison
    fig, ax = plt.subplots(figsize=figsize)
    plot_dem_difference(ax, dem1, dem2, add_text=False)
    ax.set_ylim(-1, element["num_rounds"] + 1)
    fig.tight_layout()
    fig.savefig(data_dir / f"dem_{NOISE_NAME_1}_vs_{NOISE_NAME_2}.pdf", format="pdf")
    plt.close()
