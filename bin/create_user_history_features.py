import argparse

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.features.user_history_features import get_user_history_features, get_user_last_item_stats
from lib.utils import read_data

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

    features = get_user_history_features(train_data, val_or_test_data, artists)
    last_item_popularity = get_user_last_item_stats(train_data, val_or_test_data)
    features.merge(last_item_popularity).to_parquet(args.out)
