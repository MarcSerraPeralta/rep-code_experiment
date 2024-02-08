print("Importing libraries...")
import os
import pathlib
import random

import numpy as np
import tensorflow as tf

from qrennd import Config, Layout, get_callbacks, get_model, set_coords
from rep_code.dataset import load_nn_dataset


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
anc_qubits = layout.get_qubits(role="anc")
num_anc = len(anc_qubits)

if config.model["type"] in ("ConvLSTM", "Conv_LSTM"):
    rec_features = (layout.distance + 1, layout.distance + 1, 1)
else:
    rec_features = num_anc

if config.dataset["input"] == "measurements":
    data_qubits = layout.get_qubits(role="data")
    eval_features = len(data_qubits)
else:
    eval_features = int(num_anc / 2)


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
