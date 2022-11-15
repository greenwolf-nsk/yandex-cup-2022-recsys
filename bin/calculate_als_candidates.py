import argparse

import implicit.als

from lib.candidates.als import AlsRecommender
from lib.utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--model')
parser.add_argument('--items_per_user', type=int)
parser.add_argument('--use_last_items', type=int)
parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    als = implicit.als.AlternatingLeastSquares()
    als = als.load(args.model)
    model: AlsRecommender = AlsRecommender(als)
    data = [sess[-args.use_last_items:] for sess in read_data(args.data)]
    candidates_df = model.recommend(data, args.items_per_user)
    candidates_df.to_parquet(args.out)


