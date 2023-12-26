from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit

FILE_NAMES = [...]

for f_name in FILE_NAMES:
    # calibrate readout
    iq_readout_data = xr.load_dataset("...{f_name}/readout_calibration_IQ.nc")
    shots_0, shots_1 = (iq_readout_data.sel(state=i) for i in range(2))
    cla_qubit_params = TwoStateLinearClassifierFit(shots_0, shots_1)
    np.save(
        "...{f_name}/readout_calibration_TwoStateLinearClassifierFit.npy",
        cla_qubit_params,
    )
