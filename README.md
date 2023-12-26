# rep-code_experiment
Analysis of the repetition code experiment

## Code structure

- `rep_code`: functions and objects specific to the repetition code experiment (layout, stim circuit, heralded init, defect calculation?, function to convert raw data to xarrays...). The other things that are common to other experiments should be placed in `qec_util`. 
- `dataset`: contains the data and script to convert the data into xarrays
- `lut_decoder`: has the LUT code and the script to run it
- `mwpm_decoder`: contains script to obtain the DEM and script to run MWPM decoder
- `nn_decoder`: contrains scripts to train, analyse training, and decode 

*Note: maybe group all the decoders into a single folder `decoders`?*

## Data structure

QEC DATA:
```
xr.Dataset(
    data_vars=dict(
        anc_meas=(
            ["shot", "qec_round", "anc_qubit", "iq"],
            anc_meas.astype(float),
        ),
        data_meas=(
            ["shot", "data_qubit", "iq"], 
            data_meas.astype(float)),
        ideal_anc_meas=(
            ["qec_round", "anc_qubit"],
            ideal_anc_meas.astype(bool),
        ),
        ideal_data_meas=(
            ["data_qubit"],
            ideal_data_meas.astype(bool),
        ),
        data_init=(
            ["data_qubit"],
            ideal_data_meas.astype(bool),
        ),
        heralded_init=(
            ["shot", "data_qubit", "iq"], 
            heralded_init.astype(float)),
        )
    ),
    coords=dict(
        data_qubit=data_qubit,
        anc_qubit=anc_qubit,
        shot=shot,
        qec_round=qec_round,
        rot_basis=rot_basis,
        iq=["I", "Q"],
    ),
    attrs=dict(
        ...
        data_type="IQ",
    ),
)
```

READOUT CALIBRATION DATA:
```
xr.DataArray(
    data=data,
    coords=dict(
        state=[0,1,2],
        shot=shot,
        qubit=qubit,
        iq=["I", "Q"],
    )
    attrs=dict(
        ...
        data_type="IQ",
    ),
)
```