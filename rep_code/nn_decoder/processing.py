from typing import Optional, Tuple

import xarray as xr

from qrennd.datasets.preprocessing import (
    get_syndromes,
    get_defects,
    get_defect_probs,
    get_final_defects,
    get_final_defect_probs,
)


def to_defect_probs_leakage_IQ(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    classifiers: dict,
    digitization: Optional[dict] = {"data": True, "anc": True},
    leakage: Optional[dict] = {"data": False, "anc": False},
):
    """
    Preprocess dataset to generate the probability of defect
    based on the soft outcomes and the logical errors.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas: [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas: [qec_cycle, anc_qubit]
        - data_meas: [shot, data_qubit]
        - anc_leakage_flag: [shots, qec_cycle, anc_qubit]
        - data_leakage_flag: [shot, data_qubit]
        - idea_data_meas: [data_qubit]
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    digitization
        Flag for digitizing the defect probability
    """
    anc_probs, data_probs = get_state_probs_IQ(dataset, classifiers)

    ideal_syndromes = get_syndromes(dataset.ideal_anc_meas)
    ideal_defects = get_defects(ideal_syndromes)
    defect_probs = get_defect_probs(anc_probs, ideal_defects)

    ideal_proj_syndrome = (dataset.ideal_data_meas @ proj_mat) % 2
    ideal_final_defects = get_final_defects(ideal_syndromes, ideal_proj_syndrome)
    final_defect_probs = get_final_defect_probs(
        anc_probs,
        data_probs,
        ideal_final_defects=ideal_final_defects,
        proj_mat=proj_mat,
    )

    data_meas = data_probs.sel(state=1) > data_probs.sel(state=0)
    data_meas = data_meas.transpose("shot", "data_qubit")
    data_flips = data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    if digitization["anc"]:
        defect_probs = defect_probs > 0.5
    if digitization["data"]:
        final_defect_probs = final_defect_probs > 0.5

    # add leakage outcomes
    rec_inputs, eval_inputs = defect_probs, final_defect_probs
    if leakage["anc"]:
        anc_leakage_flag = dataset.anc_leakage_flag
        rec_inputs = [defect_probs, anc_leakage_flag]
    if leakage["data"]:
        data_leakage_flag = dataset.data_leakage_flag
        eval_inputs = [final_defect_probs, data_leakage_flag]

    return (
        rec_inputs,
        eval_inputs,
        log_errors,
    )


def get_state_probs_IQ(
    dataset: xr.Dataset, classifiers: dict
) -> Tuple[xr.DataArray, xr.DataArray]:
    # data qubits
    probs_0_list = []
    for qubit in dataset.data_qubit:
        cla = classifiers[qubit.values.item()]
        outcomes = dataset.data_meas.sel(data_qubit=qubit).transpose(..., "iq")
        # DecayClassifier does not have "pdf_0"
        probs = xr.apply_ufunc(
            lambda x: cla.pdf_0_projected(cla.project(x)),
            outcomes,
            input_core_dims=[["iq"]],
            output_dtypes=[float],
        )
        probs_0_list.append(probs)
    probs_0_list = xr.concat(probs_0_list, dim="data_qubit")

    probs_1_list = []
    for qubit in dataset.data_qubit:
        cla = classifiers[qubit.values.item()]
        outcomes = dataset.data_meas.sel(data_qubit=qubit)
        # DecayClassifier does not have "pdf_0"
        probs = xr.apply_ufunc(
            lambda x: cla.pdf_0_projected(cla.project(x)),
            outcomes,
            input_core_dims=[["iq"]],
            output_dtypes=[float],
        )
        probs_1_list.append(probs)
    probs_1_list = xr.concat(probs_1_list, dim="data_qubit")
    data_probs = xr.concat([probs_0_list, probs_1_list], dim="state")

    # ancilla qubits
    probs_0_list = []
    for qubit in dataset.anc_qubit:
        cla = classifiers[qubit.values.item()]
        outcomes = dataset.anc_meas.sel(anc_qubit=qubit)
        # DecayClassifier does not have "pdf_0"
        probs = xr.apply_ufunc(
            lambda x: cla.pdf_0_projected(cla.project(x)),
            outcomes,
            input_core_dims=[["iq"]],
            output_dtypes=[float],
        )
        probs_0_list.append(probs)
    probs_0_list = xr.concat(probs_0_list, dim="anc_qubit")

    probs_1_list = []
    for qubit in dataset.anc_qubit:
        cla = classifiers[qubit.values.item()]
        outcomes = dataset.anc_meas.sel(anc_qubit=qubit)
        # DecayClassifier does not have "pdf_0"
        probs = xr.apply_ufunc(
            lambda x: cla.pdf_0_projected(cla.project(x)),
            outcomes,
            input_core_dims=[["iq"]],
            output_dtypes=[float],
        )
        probs_1_list.append(probs)
    probs_1_list = xr.concat(probs_1_list, dim="anc_qubit")
    anc_probs = xr.concat([probs_0_list, probs_1_list], dim="state")

    anc_probs = anc_probs / anc_probs.sum(dim="state")
    data_probs = data_probs / data_probs.sum(dim="state")

    anc_probs = anc_probs.transpose("state", "shot", "qec_round", "anc_qubit")
    data_probs = data_probs.transpose("state", "shot", "data_qubit")

    return anc_probs, data_probs
