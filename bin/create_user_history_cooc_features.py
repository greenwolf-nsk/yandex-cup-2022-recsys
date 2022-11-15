import argparse

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.features.user_history_cooc_features import get_user_history_cooc_features
from lib.utils import read_data, create_df_from_sessions

parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--cooc_stats')
parser.add_argument('--val_or_test_data')
parser.add_argument('--history_last_items', type=int)

parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    val_or_test_data = read_data(args.val_or_test_data)
    candidates = pd.read_parquet(args.candidates)[['user_id', 'item_id']]
    print(candidates.shape)
    cooc_stats = pd.read_parquet(args.cooc_stats)
    val_or_test_history = create_df_from_sessions([
        sess[-args.history_last_items:]
        for sess in val_or_test_data
    ])


    features = get_user_history_cooc_features(
        candidates,
        val_or_test_history,
        cooc_stats
    )
    del candidates
    del cooc_stats
    del val_or_test_history
    features.to_parquet(args.out)
