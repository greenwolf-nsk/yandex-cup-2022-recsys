import argparse

from lib.candidates.popular import train_popular_model
from lib.utils import read_data, save_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--last_popular_items', type=int)
parser.add_argument('--model')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    model = train_popular_model([x[-args.last_popular_items:] for x in data])
    save_pickle(model, args.model)




