def to_defect_probs_leakage_IQ(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    digitization: Optional[dict] = {"data": False, "anc": False},
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
    anc_probs, data_probs = get_state_probs_IQ(dataset)

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
