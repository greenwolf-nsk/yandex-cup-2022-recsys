import argparse

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.features.user_artist_features import get_user_artist_stats
from lib.utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--artists')
parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    artists = pd.read_parquet(args.artists)
    features = get_user_artist_stats(data, artists)
    features.to_parquet(args.out)
