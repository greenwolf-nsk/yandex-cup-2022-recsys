import os

N_ITEMS = 483275
N_ARTISTS = 56134
USE_CUDF = os.environ.get('USE_CUDF', 0)
DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
if USE_CUDF and DEVICES:
    print(f'using cuda, devices: {DEVICES}')

import pandas as pd

if USE_CUDF:
    import cudf
    pd = cudf

