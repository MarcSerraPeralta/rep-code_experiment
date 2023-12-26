import numpy as np
import xarray as xr

from rep_code.readout import get_hard_outcomes, ps_heralded_init
from rep_code.defects import get_syndromes, get_defects, get_final_defects

FILE_NAMES = [...]  # depend on basis and initial bitstring
LIST_NUM_ROUNDS = [...]  # list of the total number of QEC rounds
CLASSIFIER = "TwoStateLinearClassifierFit"

for f_name in FILE_NAMES:
    for num_rounds in LIST_NUM_ROUNDS:
        # load readout calibration and iq data
        cla_qubit_params = np.load("...{f_name}/readout_calibration_{CLASSIFIER}.npy")
        iq_data = xr.load_dataset("...{f_name}/measurements_r{num_rounds}.nc")

        # get hard outcomes by calibrating readout
        iq_vars = ["anc_meas", "data_meas", "heralded_init"]
        outcomes = get_hard_outcomes(iq_data, cla_qubit_params, iq_vars)

        # post select based on heralded meas
        heralded_var = "heralded_init"
        outcomes = ps_heralded_init(outcomes, heralded_var=heralded_var)
        outcomes = outcomes.drop_vars([heralded_var])

        # get defects
        proj_mat = layout.projection_matrix(
            type="x_type" if outcomes.rot_basis else "z_type"
        )
        frame = (outcomes.data_init @ layout.projection_matrix()) % 2
        anc_flips = dataset.anc_meas ^ dataset.ideal_anc_meas
        data_flips = dataset.data_meas ^ dataset.ideal_data_meas
        syndromes = get_syndromes(anc_flips)
        proj_syndrome = (data_flips @ proj_mat) % 2

        defects = get_defects(syndromes, frame)
        final_defects = get_final_defects(syndromes, proj_syndrome)

        # get Pij matrix
        didj = get_pij_matrix(defects)
        di = get_ave_defects(defects)

        # get decoding graph including
        # edges and boundary edges
        dem = get_decoding_graph(didj=didj, di=di)

        # store dem
        ...
