from typing import Generator, List, Optional

import pathlib

import numpy as np
import xarray as xr

from qec_util import Layout
from iq_readout import two_state_classifiers, three_state_classifiers
from qrennd.configs import Config
from qrennd.datasets.sequences import RaggedSequence, Sequence
from qrennd.datasets.preprocessing import to_model_input

from .processing import to_defect_probs_leakage_IQ


def load_nn_dataset(
    config: Config,
    layout: Layout,
    dataset_name: str,
    concat: Optional[bool] = True,
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
        two_state_classifiers = get_classifiers(
            config.dataset["two_state_classifier"],
            config.experiment_dir / "readout_calibration",
            num_states=2,
        )
        three_state_classifiers = get_classifiers(
            config.dataset["three_state_classifier"],
            config.experiment_dir / "readout_calibration",
            num_states=3,
        )
        processed_gen = (
            to_defect_probs_leakage_IQ(
                dataset,
                proj_mat=proj_matrix,
                two_state_classifiers=two_state_classifiers,
                three_state_classifiers=three_state_classifiers,
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

    if concat:
        return RaggedSequence.from_generator(input_gen, batch_size, False)

    sequences = (Sequence(*tensors, batch_size, False) for tensors in input_gen)
    return sequences


def get_classifiers(
    classifier_name: str,
    path_to_params: pathlib.Path,
    num_states: int,
) -> dict:
    if num_states == 2:
        module = two_state_classifiers
    elif num_states == 3:
        module = three_state_classifiers

    if classifier_name == "GaussMixLinearClassifier":
        classifier = module.GaussMixLinearClassifier
    elif classifier_name == "DecayLinearClassifier":
        classifier = module.DecayLinearClassifier
    elif classifier_name == "GaussMixClassifier":
        classifier = module.GaussMixClassifier
    else:
        raise ValueError(
            "Classifier name must be GaussMixLinearClassifier, DecayLinearClassifier, or GaussMixClassifier;"
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
    **kargs,
) -> Generator:
    for num_rounds in rounds:
        for state in states:
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                num_rounds=num_rounds,
                **kargs,
            )
            dataset = xr.open_dataset(
                dataset_dir / experiment / f"iq_data_{dataset_name}.nc"
            )
            yield dataset
