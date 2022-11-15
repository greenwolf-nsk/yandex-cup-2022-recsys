import argparse
import os
import random

import pandas as pd

from lib.utils import read_data, save_sequence_data_dir_factory

random.seed(33)
parser = argparse.ArgumentParser()
parser.add_argument('--val_size')
parser.add_argument('--sample_size')
parser.add_argument('--raw_data_dir')
parser.add_argument('--num_target_events', type=int)
parser.add_argument('--processed_data_dir')


def train_val_split(data: list, val_size: int) -> tuple:
    random.shuffle(data)
    return data[val_size:], data[:val_size]


def targets_split(data: list, num_events: int = 1) -> tuple:
    return [seq[:-num_events] for seq in data], [seq[-num_events:] for seq in data]


def sample(data: list, sample_size: int) -> list:
    random.shuffle(data)
    return data[:sample_size]


if __name__ == '__main__':
    args = parser.parse_args()
    sample_size = int(args.sample_size)
    val_size = int(args.val_size)
    raw_data_dir = args.raw_data_dir
    processed_data_dir = args.processed_data_dir
    sample_data_dir = os.path.join(processed_data_dir, 'sample')
    if not os.path.exists(sample_data_dir):
        os.makedirs(sample_data_dir, exist_ok=True)

    save_sequence_data = save_sequence_data_dir_factory(args.processed_data_dir)
    save_sequence_data_sample = save_sequence_data_dir_factory(sample_data_dir)
    train_raw = read_data(os.path.join(raw_data_dir, 'train'))
    test = read_data(os.path.join(raw_data_dir, 'test'))
    save_sequence_data(test, 'test')
    train, val = train_val_split(train_raw, val_size)
    train_sample = sample(train, sample_size)

    save_sequence_data(train, 'train')
    save_sequence_data_sample(train_sample, 'train')

    test_sample = sample(test, sample_size)
    save_sequence_data_sample(test_sample, 'test')

    val_no_targets, val_targets = targets_split(val, args.num_target_events)
    save_sequence_data(val_no_targets, 'val')
    save_sequence_data(val_targets, 'val_targets')

    val_sample = sample(val, sample_size)
    val_sample_no_targets, val_sample_targets = targets_split(val_sample, args.num_target_events)
    save_sequence_data_sample(val_sample_no_targets, 'val')
    save_sequence_data_sample(val_sample_targets, 'val_targets')

    train_fullest = train + val_no_targets + test
    train_fullest_sample = sample(train_fullest, sample_size)
    save_sequence_data(train_fullest, 'train_fullest')
    save_sequence_data_sample(train_fullest_sample, 'train_fullest')

    artists = pd.read_csv(os.path.join(raw_data_dir, 'track_artists.csv'))
    artists.columns = ['item_id', 'artist_id']
    artists.to_parquet(os.path.join(processed_data_dir, 'artists.pq'))
