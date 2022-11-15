import argparse

from const import USE_CUDF
from lib.candidates.similar import SimilarRecommender
from lib.features.user_history_als_features import get_candidates_lists_from_df
from lib.features.user_history_similarity_features import create_similarity_features, SIMILARITY_AGGS_MAP
from lib.utils import read_data, load_pickle

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--similarity_model')
parser.add_argument('--val_or_test_data')
parser.add_argument('--history_last_items', type=int)

parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    model: SimilarRecommender = load_pickle(args.similarity_model)
    similarity_model = model.implicit_model
    sorted_candidates = pd.read_parquet(args.candidates)[['user_id', 'item_id']].sort_values('user_id')
    candidates_list = get_candidates_lists_from_df(sorted_candidates)
    val_or_test_data = [
        x[-args.history_last_items:]
        for x in read_data(args.val_or_test_data)
    ]
    features = create_similarity_features(
        similarity_model,
        candidates_list,
        val_or_test_data,
        SIMILARITY_AGGS_MAP,
    )
    features.write_parquet(args.out)
