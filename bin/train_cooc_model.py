import argparse

from lib.utils import read_data, save_pickle
from lib.candidates.coocurence import train_cooc_model

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--model')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    model = train_cooc_model(data)
    save_pickle(model, args.model)




