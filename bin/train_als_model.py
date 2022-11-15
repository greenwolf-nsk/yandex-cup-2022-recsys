import argparse

from lib.candidates.als import train_als_model
from lib.utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--factors', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--model')


if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    model = train_als_model(data, factors=args.factors, iterations=args.iterations)
    model.save(args.model)
