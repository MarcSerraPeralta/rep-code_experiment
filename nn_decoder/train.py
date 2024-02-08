print("Importing libraries...")
import os
import pathlib
import random

import numpy as np
import tensorflow as tf

from qrennd import Config, Layout, get_callbacks, get_model, set_coords
from rep_code.nn_decoder import load_nn_dataset


# Parameters
LAYOUT_FILE = "rep_code_layout.yaml"
CONFIG_FILE = "fist_try_config.yaml"

USERNAME = os.environ.get("USER")
SCRATH_DIR = pathlib.Path(f"/scratch/{USERNAME}")

DATA_DIR = SCRATH_DIR / "projects" / "20231220-repetition_code_dicarlo_lab" / "nn_data"
OUTPUT_DIR = (
    SCRATH_DIR / "projects" / "20231220-repetition_code_dicarlo_lab" / "nn_output"
)

#########################

print("Running script...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load setup objects
CONFIG_DIR = pathlib.Path.cwd() / "configs"
config = Config.from_yaml(
    filepath=CONFIG_DIR / CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)
config.log_dir.mkdir(exist_ok=True, parents=True)
config.checkpoint_dir.mkdir(exist_ok=True, parents=True)

LAYOUT_DIR = DATA_DIR / config.experiment / "config"
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)

# set random seed for tensorflow, numpy and python
if config.seed is None:
    config.seed = np.random.randint(1e15)
random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

config.to_yaml(config.run_dir / "config.yaml")

# load datasets
train_data = load_nn_dataset(config=config, layout=layout, dataset_name="train")
val_data = load_nn_dataset(config=config, layout=layout, dataset_name="val")

# input features
num_data = len(layout.get_qubits(role="data"))
num_anc = len(layout.get_qubits(role="anc"))
leakage = config.dataset.get("leakage")

rec_features = num_anc
eval_features = num_data

if config.dataset["input"] == "experimental_data":
    if leakage["anc"]:
        rec_features += num_anc
    if leakage["data"]:
        eval_features += num_data
else:
    raise ValueError(f"config.dataset.input is not correct: {config.dataset['input']}")


model = get_model(
    rec_features=rec_features,
    eval_features=eval_features,
    config=config,
)
callbacks = get_callbacks(config)

# train model
history = model.fit(
    train_data,
    validation_data=val_data,
    batch_size=config.train["batch_size"],
    epochs=config.train["epochs"],
    callbacks=callbacks,
    shuffle=True,
    verbose=0,
)

# save model
model.save(config.checkpoint_dir / "final_weights.hdf5")
