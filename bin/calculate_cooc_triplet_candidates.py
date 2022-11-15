import argparse

from lib.utils import read_data, load_pickle, polars_outer_merge
from lib.candidates.coocurence import CoocurenceTripletRecommender, train_triplet_model

parser = argparse.ArgumentParser()
parser.add_argument('--train_data')
parser.add_argument('--val_or_test_data')
parser.add_argument('--model')
parser.add_argument('--item_indices', nargs="+", type=int)
parser.add_argument('--out')


if __name__ == '__main__':
    args = parser.parse_args()
    train_data = read_data(args.train_data)
    cooc_triples = train_triplet_model(train_data)
    models = [
        CoocurenceTripletRecommender(cooc_triples, idx)
        for idx in args.item_indices
    ]
    val_or_test_data = read_data(args.val_or_test_data)
    candidates_df = models[0].recommend(val_or_test_data)
    for model in models[1:]:
        df = model.recommend(val_or_test_data)
        candidates_df = polars_outer_merge(candidates_df, df)

    candidates_df.to_pandas().to_parquet(args.out)



