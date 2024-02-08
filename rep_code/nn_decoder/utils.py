from typing import Generator, List

import pathlib

import numpy as np
import xarray as xr

from qec_util import Layout
from iq_readout.two_state_classifiers import *
from qrennd.configs import Config
from qrennd.datasets.sequences import RaggedSequence
from qrennd.datasets.preprocessing import to_model_input

from .processing import to_defect_probs_leakage_IQ


def load_nn_dataset(
    config: Config,
    layout: Layout,
    dataset_name: str,
):
    batch_size = config.train["batch_size"]
    model_type = config.model["type"]
    experiment_name = config.dataset["folder_format_name"]

    rot_basis = config.dataset["rot_basis"]
    basis = "X" if rot_basis else "Z"

    dataset_dir = config.experiment_dir
    dataset_params = config.dataset[dataset_name]

    dataset_gen = dataset_generator(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        basis=basis,
        **dataset_params,
    )
    proj_matrix = layout.projection_matrix("x_type")

    # Convert to desired input
    input_type = config.dataset["input"]
    if input_type == "experimental_data":
        digitization = config.dataset.get("digitization")
        leakage = config.dataset.get("leakage")
        classifiers = get_classifiers(
            config.dataset["classifier"], config.experiment_dir / "readout_calibration"
        )
        processed_gen = (
            to_defect_probs_leakage_IQ(
                dataset,
                proj_mat=proj_matrix,
                classifiers=classifiers,
                leakage=leakage,
                digitization=digitization,
            )
            for dataset in dataset_gen
        )
    else:
        raise ValueError(
            f"Unknown input data type {input_type}, the possible "
            "options are 'experimental_data'."
        )

    # Process for keras.model input
    conv_models = ("ConvLSTM", "Conv_LSTM")
    exp_matrix = layout.expansion_matrix() if (model_type in conv_models) else None
    data_type = float if input_type == "prob_defects" else bool
    input_gen = (to_model_input(*arrs, exp_matrix, data_type) for arrs in processed_gen)

    return RaggedSequence.from_generator(input_gen, batch_size, False)


def get_classifiers(classifier_name, path_to_params):
    if classifier_name == "TwoStateLinearClassifierFit":
        classifier = TwoStateLinearClassifierFit
    elif classifier_name == "DecayLinearClassifierFit":
        classifier = DecayLinearClassifierFit
    else:
        raise ValueError(
            "Classifier name must be TwoStateLinearClassifierFit or DecayLinearClassifierFit, "
            f"but {classifier_name} was given"
        )

    cla_params = np.load(
        path_to_params / f"{classifier_name}_params_ps.npy", allow_pickle=True
    ).item()
    classifiers = {q: classifier().load(p) for q, p in cla_params.items()}

    return classifiers


def dataset_generator(
    dataset_dir: pathlib.Path,
    dataset_name: str,
    experiment_name: str,
    basis: str,
    states: List[str],
    rounds: List[int],
    **args,
) -> Generator:
    for num_rounds in rounds:
        for state in states:
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                num_rounds=num_rounds,
                **args,
            )
            dataset = xr.open_dataset(
                dataset_dir / experiment / f"iq_data_{dataset_name}.nc"
            )
            yield dataset
