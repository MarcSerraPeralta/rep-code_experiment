# %%
import pathlib
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import yaml
from lmfit import Model

from qrennd.utils.analysis import lmfit_par_to_ufloat

from rep_code.decoding import (
    plot_fidelity_exp,
    plot_fidelity_fit,
    get_error_rate,
    plot_error_prob_fit,
    plot_error_prob_exp,
)

# %%
EXP_NAME = "20240119_initial_data_d5"
MODEL_FOLDER = "20240209-170544_lstm64x2_eval64_b256_dr0-05_lr0-001"
DATASET = "test.nc"

TITLE = f"{EXP_NAME}\n{MODEL_FOLDER} ({DATASET})"
OUTPUT_DIR = pathlib.Path(EXP_NAME) / MODEL_FOLDER
OUTPUT_NAME = DATASET.replace(".nc", "")

FIT = True

# %%
DIR = pathlib.Path.cwd() / "evaluation" / "output"
DATA_DIR = pathlib.Path.cwd() / "evaluation" / "data"

with open(DATA_DIR / EXP_NAME / "config_data.yaml", "r") as file:
    config_data = yaml.safe_load(file)
distance = config_data["string_data_options"]["distance"]

# %%
fig, ax = plt.subplots(figsize=(7, 5))

# %%
EXP_DIR_ = EXP_NAME
RUN_DIR_ = MODEL_FOLDER
COLOR = "blue"
LABEL_DATA = "NN"

log_fid = xr.load_dataset(DIR / EXP_DIR_ / RUN_DIR_ / DATASET)
num_rounds = log_fid.qec_round.values
log_errors = log_fid.transpose("qec_round", ...).errors.values

fig, ax = plt.subplots()

if FIT:
    error_rate, r0 = get_error_rate(num_rounds, log_errors, distance, return_r0=True)
    plot_fidelity_fit(
        ax,
        num_rounds,
        error_rate.nominal_value,
        r0.nominal_value,
        distance,
        color=COLOR,
        label=f"$\\epsilon_L$ = {error_rate*100}%",
        linestyle="-",
    )

plot_fidelity_exp(
    ax,
    num_rounds,
    log_errors,
    color=COLOR,
    linestyle="",
    fmt=".",
    label=LABEL_DATA,
)

ax.legend(loc="upper right")
fig.tight_layout()
for format_ in ["svg", "pdf", "png"]:
    fig.savefig(DIR / OUTPUT_DIR / f"{OUTPUT_NAME}.{format_}", format=format_)

plt.close()

fig, ax = plt.subplots()

if FIT:
    plot_error_prob_fit(
        ax,
        num_rounds,
        error_rate.nominal_value,
        r0.nominal_value,
        distance,
        color=COLOR,
        label=f"$\\epsilon_L$ = {error_rate*100}%",
        linestyle="-",
    )

plot_error_prob_exp(
    ax,
    num_rounds,
    log_errors,
    color=COLOR,
    linestyle="",
    fmt=".",
    label=LABEL_DATA,
)

ax.legend(loc="upper right")
fig.tight_layout()
for format_ in ["svg", "pdf", "png"]:
    fig.savefig(DIR / OUTPUT_DIR / f"{OUTPUT_NAME}_log-scale.{format_}", format=format_)

plt.close()
