import pathlib
import os

import yaml

RAW_DATA_DIR = pathlib.Path("/tudelft.net/staff-umbrella/repcode/")

EXP_NAMES = next(os.walk(RAW_DATA_DIR))[1]  # list only directories

for exp_name in EXP_NAMES:
    run_names = next(os.walk(RAW_DATA_DIR / exp_name))[1]

    for run_name in run_names:
        run_dir = RAW_DATA_DIR / exp_name / run_name

        metadata = {
            "time_units": "ns",
            "setup": [
                dict(
                    cz_error_prob=0.01,
                    sq_error_prob=0.1,
                    meas_error_prob=0.01,
                    reset_error_prob=0.0,
                    assign_error_flag=True,
                    assign_error_prob=0.01,
                    T1=20000,
                    T2=21000,
                ),
            ],
            "gate_durations": dict(
                X=30,
                H=30,
                CZ=40,
                R=1000,
                M=500,
                X_ECHO=500,
            ),
        }

        with open(run_dir / "device_characterization.yaml", "w") as outfile:
            yaml.dump(metadata, outfile, default_flow_style=False)
