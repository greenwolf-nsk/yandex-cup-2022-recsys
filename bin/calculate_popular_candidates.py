import argparse

from lib.candidates.popular import PopularRecommender
from lib.utils import read_data, load_pickle

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data')
parser.add_argument('-m', '--model')
parser.add_argument('-o', '--out')

if __name__ == '__main__':
    args = parser.parse_args()
    model: PopularRecommender = load_pickle(args.model)
    data = read_data(args.data)
    candidates_df = model.recommend(data)
    candidates_df.to_parquet(args.out)



