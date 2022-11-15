import argparse

from lib.features.item_features import get_items_popularity_stats
from lib.utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--offsets', nargs='+', type=int)
parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    features = get_items_popularity_stats(data, offsets=args.offsets)
    features.to_parquet(args.out)
