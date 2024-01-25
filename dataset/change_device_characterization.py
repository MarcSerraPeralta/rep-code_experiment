import os
import yaml


def append_empty_line_to_files(root_directory, new_config):
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename == "device_characterization.yaml":
                file_path = os.path.join(root, filename)
                with open(file_path, "w") as file:
                    yaml.dump(new_config, file, default_flow_style=False)


directory_path = (
    "/scratch/marcserraperal/projects/20231220-repetition_code_dicarlo_lab/data"
)

with open("configs/device_characterization.yaml", "r") as file:
    new_config = yaml.safe_load(file)

append_empty_line_to_files(directory_path, new_config)
