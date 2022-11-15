import argparse

import implicit

from const import pd

from lib.features.user_history_als_features import create_als_similarity_features, get_candidates_lists_from_df, \
    DEFAULT_AGGS_MAP
from lib.utils import read_data


parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--als_model')
parser.add_argument('--val_or_test_data')
parser.add_argument('--history_last_items', type=int)

parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    als = implicit.als.AlternatingLeastSquares()
    als = als.load(args.als_model)
    sorted_candidates = pd.read_parquet(args.candidates)[['user_id', 'item_id']].sort_values('user_id')
    candidates_list = get_candidates_lists_from_df(sorted_candidates)
    val_or_test_data = [
        x[-args.history_last_items:]
        for x in read_data(args.val_or_test_data)
    ]
    features = create_als_similarity_features(
        als.item_factors.to_numpy(),
        candidates_list,
        val_or_test_data,
        DEFAULT_AGGS_MAP,
    )
    features.write_parquet(args.out)
