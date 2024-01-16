import os


def rename_directories(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            original_path = os.path.join(dirpath, dirname)
            new_path = os.path.join(
                dirpath,
                dirname.replace(
                    "_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29",
                    "",
                ),
            )

            if original_path != new_path:
                os.rename(original_path, new_path)
                print(f"Renamed: {original_path} -> {new_path}")

        for filename in filenames:
            original_path = os.path.join(dirpath, filename)
            new_filename = filename.replace(
                "_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29",
                "",
            )
            new_path = os.path.join(dirpath, new_filename)

            if original_path != new_path:
                os.rename(original_path, new_path)
                print(f"Renamed: {original_path} -> {new_path}")


if __name__ == "__main__":
    directory_path = "."
    rename_directories(directory_path)
