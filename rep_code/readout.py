import numpy as np
import xarray as xr

# from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit


def get_readout_classifiers(data: xr.DataArray) -> dict:
    """
    Fits the TwoStateLinearClassifierFit to all qubits
    given the IQ readout calibration data

    Parameters
    ----------
    data
        Dataset with all the readout calibration IQ data.
        It must contain the following coordinates:
        state=[0,1,2], shot=shots, qubit=qubits, iq=["I", "Q"],

    Returns
    -------
    cla_qubit_params
        Dictionary with the parameters to initialize the
        readout classifier using `load` function for each qubit.
        {q:TwoStateLinearClassifierFit.params() for q in qubits}
    """
    # check input
    assert set(data.coords) == set(["state", "shot", "qubit", "iq"])
    assert set(data.state.values) >= set([0, 1])
    assert set(data.iq.values) == set(["I", "Q"])

    cla_qubit_params = {}
    for qubit in data.qubit:
        data_q = data.sel(qubit=qubit)
        cla = TwoStateLinearClassifierFit().fit(
            data_q.sel(state=0), data_q.sel(state=1)
        )
        cla_qubit_params[qubit.values] = cla.params()

    return cla_qubit_params


def get_hard_outcomes(
    iq_data: xr.Dataset, cla_qubit_params: dict, iq_vars: list
) -> xr.Dataset:
    """
    Calculates the hard outcomes from the given iq_data
    using the cla_qubit_params and TwoStateLinearClassifierFit.

    Parameters
    ----------
    iq_data
        Dataset with IQ data to digitize
    cla_qubit_params
        Dictionary with the parameters to initialize the
        readout classifier using `load` function for each qubit.
        {q:TwoStateLinearClassifierFit.params() for q in qubits}
    iq_vars
        Variables in `iq_data` with IQ data to digitize.
        They have to be from the following set:
        ["anc_meas", "data_meas", "heralded_init"].

    Returns
    -------
    iq_data
        Same dataset but with the `iq_vars` digitized
    """
    # check inputs
    assert set(iq_vars) <= set(["anc_meas", "data_meas", "heralded_init"])

    if "anc_meas" in iq_vars:
        anc_outcomes = []
        for qubit in iq_data.anc_qubit:
            iq = (
                iq_data.anc_meas.sel(anc_qubit=qubit)
                .transpose("shot", "qec_round", "iq")
                .values
            )
            cla = TwoStateLinearClassifierFit().load(cla_qubit_params[qubit])
            out = cla.predict(iq)
            anc_outcomes.append(out)
        iq_data.anc_meas = xr.DataArray(
            data=anc_outcomes,
            coords=dict(
                anc_qubit=iq_data.anc_qubit,
                shot=iq_data.shot,
                qec_round=iq_data.qec_round,
            ),
        ).transpose("shot", "qec_round", "anc_qubit")

    if "data_meas" in iq_vars:
        data_outcomes = []
        for qubit in iq_data.data_qubit:
            iq = iq_data.data_meas.sel(data_qubit=qubit).transpose("shot", "iq").values
            cla = TwoStateLinearClassifierFit().load(cla_qubit_params[qubit])
            out = cla.predict(iq)
            data_outcomes.append(out)
        iq_data.data_meas = xr.DataArray(
            data=data_outcomes,
            coords=dict(
                data_qubit=iq_data.data_qubit,
                shot=iq_data.shot,
            ),
        ).transpose("shot", "data_qubit")

    if "heralded_init" in iq_vars:
        data_outcomes = []
        for qubit in iq_data.data_qubit:
            iq = (
                iq_data.heralded_init.sel(data_qubit=qubit)
                .transpose("shot", "iq")
                .values
            )
            cla = TwoStateLinearClassifierFit().load(cla_qubit_params[qubit])
            out = cla.predict(iq)
            data_outcomes.append(out)
        iq_data.heralded_init = xr.DataArray(
            data=data_outcomes,
            coords=dict(
                data_qubit=iq_data.data_qubit,
                shot=iq_data.shot,
            ),
        ).transpose("shot", "data_qubit")

    return


def ps_heralded_init(outcomes: xr.Dataset, heralded_var: str):
    """
    Post select shots based on heralded initialization

    Parameters
    ----------
    outcomes
        Dataset in which to post select the shots
    heralded_var
        Variable in `outcomes` that has the heralded initialization
        outcomes. It must have coordinates: "shot", "data_qubit"
    """
    heralded_init = outcomes[heralded_var].transpose("shot", "data_qubit").values
    indices = np.where(np.sum(heralded_init, axis=1) != 0)[0]
    outcomes = outcomes.isel(shot=indices)
    assert (outcomes[heralded_var].values == 0).all()

    return outcomes
