from typing import Dict, Iterator

from itertools import product
import pathlib

import xarray as xr


def sequence_generator(string_data: Dict) -> Iterator:
    # split string variables depending if they are a list or not
    single_data = {}
    list_values = []
    list_keys = []
    for k, v in string_data.items():
        if isinstance(v, list):
            list_keys.append(k)
            list_values.append(v)
        else:
            single_data[k] = v

    for combination in product(*list_values):
        dict_data = {k: v for k, v in zip(list_keys, combination)}
        dict_data.update(single_data)
        yield dict_data
