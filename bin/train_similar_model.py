import argparse

from lib.candidates.similar import train_similar_model
from lib.utils import read_data, save_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--recommender')
parser.add_argument('--model')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    model = train_similar_model(data, args.recommender)
    save_pickle(model, args.model)




