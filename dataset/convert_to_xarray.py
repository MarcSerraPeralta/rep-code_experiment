from .lib import get_iq_data

FILE_NAMES = [...]  # depend on basis and initial bitstring
LIST_NUM_ROUNDS = [...]  # list of the total number of QEC rounds

for f_name in FILE_NAMES:
    # get IQ data with "API" for DiCarlo lab
    iq_data, iq_readout_data = get_iq_data(f_name)
    for num_rounds in LIST_NUM_ROUNDS:
        iq_data[num_rounds].to_netcdf("...{f_name}/measurements_r{num_rounds}.nc")
    iq_readout_data.to_netcdf("...{f_name}/readout_calibration_IQ.nc")
