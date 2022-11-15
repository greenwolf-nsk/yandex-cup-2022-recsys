import argparse

from lib.utils import read_data, load_pickle, polars_left_merge, polars_outer_merge
from lib.candidates.coocurence import CoocurenceRecommender

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--model')
parser.add_argument('--item_indices', nargs="+", type=int)
parser.add_argument('--num_recs', type=int)
parser.add_argument('--out')


if __name__ == '__main__':
    args = parser.parse_args()
    cooc_pairs: dict = load_pickle(args.model)
    models = [
        CoocurenceRecommender(cooc_pairs, idx, args.num_recs)
        for idx in args.item_indices
    ]
    data = read_data(args.data)
    candidates_df = models[0].recommend(data)
    for model in models[1:]:
        df = model.recommend(data)
        candidates_df = polars_outer_merge(candidates_df, df)

    candidates_df.to_pandas().to_parquet(args.out)



