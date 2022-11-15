import argparse


from lib.candidates.artist import create_last_artists_recs
from lib.utils import read_data
from const import USE_CUDF

import cudf as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_data')
parser.add_argument('--val_or_test_data')
parser.add_argument('--artists')

parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    train_data = read_data(args.train_data)
    val_or_test_data = read_data(args.val_or_test_data)
    artists = pd.read_parquet(args.artists)
    last_artist_candidates = create_last_artists_recs(train_data, val_or_test_data, artists)
    last_artist_candidates.to_parquet(args.out)


