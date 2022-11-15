import argparse

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.features.artist_features import get_artists_popularity_stats, get_artist_item_stats
from lib.utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--artists')
parser.add_argument('--offsets', nargs='+', type=int)
parser.add_argument('--artist_features_out')
parser.add_argument('--artist_item_features_out')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    artists = pd.read_parquet(args.artists)
    features = get_artists_popularity_stats(data, artists, offsets=args.offsets)
    artist_item_stats = get_artist_item_stats(data, artists)
    features.to_parquet(args.artist_features_out)
    artist_item_stats.to_parquet(args.artist_item_features_out)
