from typing import Tuple, Optional, List

import numpy as np
import xarray as xr

from qec_util import Layout


def get_syndromes(anc_meas: xr.DataArray) -> xr.DataArray:
    if anc_meas.meas_reset:
        return anc_meas

    shifted_meas = anc_meas.shift(qec_round=1, fill_value=0)
    syndromes = anc_meas ^ shifted_meas
    return syndromes


def get_defects(syndromes: xr.DataArray) -> xr.DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)
    defects = syndromes ^ shifted_syn
    return defects


def get_final_defects(
    syndromes: xr.DataArray, proj_syndrome: xr.DataArray
) -> xr.DataArray:
    last_round = syndromes.qec_round.values[-1]
    anc_qubits = proj_syndrome.anc_qubit.values

    last_syndromes = syndromes.sel(anc_qubit=anc_qubits, qec_round=last_round)
    defects = last_syndromes ^ proj_syndrome
    return defects


def get_measurements(
    dataset: xr.Dataset,
    classifiers: dict,
) -> Tuple[xr.DataArray]:
    """
    TODO
    """
    # anc_meas
    anc_meas = []
    for anc_qubit in dataset.anc_qubit:
        anc_meas_q = (
            dataset.anc_meas.sel(anc_qubit=anc_qubit)
            .transpose("shot", "qec_round", "iq")
            .values
        )
        anc_name = anc_qubit.values.item()
        anc_meas.append(classifiers[anc_name].predict(anc_meas_q))
    dataset["anc_meas"] = (("anc_qubit", "shot", "qec_round"), anc_meas)

    # data_meas
    data_meas = []
    for data_qubit in dataset.data_qubit:
        data_meas_q = (
            dataset.data_meas.sel(data_qubit=data_qubit).transpose("shot", "iq").values
        )
        data_name = data_qubit.values.item()
        data_meas.append(classifiers[data_name].predict(data_meas_q))
    dataset["data_meas"] = (("data_qubit", "shot"), data_meas)

    # heralded_init
    heralded_init = []
    for qubit in dataset.qubit:
        heralded_init_q = (
            dataset.heralded_init.sel(qubit=qubit)
            .transpose("shot", "heralded_rep", "iq")
            .values
        )
        qubit_name = qubit.values.item()
        heralded_init.append(classifiers[qubit_name].predict(heralded_init_q))
    dataset["heralded_init"] = (("qubit", "shot", "heralded_rep"), heralded_init)

    return dataset.anc_meas, dataset.data_meas, dataset.heralded_init


def ps_shots_heralded(heralded_init: xr.DataArray) -> xr.DataArray:
    """
    Returns shot id to use based on
    heralded initialization post selection.
    """
    mask_bad = (heralded_init != 0).any(dim=["qubit", "heralded_rep"])
    heralded_init = heralded_init.where(~mask_bad, drop=True)
    return heralded_init.shot


def to_defects(
    anc_meas: xr.DataArray,
    data_meas: xr.DataArray,
    ideal_anc_meas: xr.DataArray,
    ideal_data_meas: xr.DataArray,
    proj_mat: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Return the defects, final defects and logical flips from
    a dataset containing the anc and data measurement outcomes
    """
    # compute defects
    # the initial frame is already present in the ideal_anc_meas
    anc_flips = anc_meas ^ ideal_anc_meas
    data_flips = data_meas ^ ideal_data_meas

    syndromes = get_syndromes(anc_flips)
    defects = get_defects(syndromes)

    proj_syndrome = (data_flips @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    log_flips = data_flips.sum(dim="data_qubit") % 2

    return defects, final_defects, log_flips


def get_defect_vector(
    defects: xr.DataArray,
    final_defects: xr.DataArray,
    anc_order: List[str],
    dim_first: str = "anc_qubit",
) -> np.ndarray:
    defects = (
        defects.transpose("shot", "qec_round", "anc_qubit")
        .sel(anc_qubit=anc_order)
        .values
    )

    n_shots, num_rounds, n_anc = defects.shape
    final_defects = (
        final_defects.transpose("shot", "anc_qubit").sel(anc_qubit=anc_order).values
    )
    final_defects = final_defects.reshape(n_shots, 1, n_anc)

    defect_vec = np.concatenate([defects, final_defects], axis=1)

    if dim_first == "anc_qubit":
        return defect_vec.reshape(n_shots, -1)
    elif dim_first == "qec_round":
        defect_vec = np.swapaxes(defect_vec, 1, 2)
        return defect_vec.reshape(n_shots, -1)
    else:
        raise ValueError(
            f"'dim_first' must be 'anc_qubit' or 'qec_round', but {dim_first} was given"
        )
