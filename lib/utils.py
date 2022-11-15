import os
import pickle
from typing import List, Iterable, Union

from numba import jit, njit, prange

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from const import N_ITEMS, USE_CUDF, pd

Sessions = List[Union[List[int], np.ndarray]]


def read_data(path: str) -> List[List[int]]:
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(list(map(int, line.strip().split(' '))))

    return data


def create_targets_df(data: List[List[int]]) -> pl.DataFrame:
    df = pl.DataFrame({
        'user_id': get_seqential_user_ids(data),
        'item_id': list(flatten(data)),
        'target': [1 / (rank + 1) for rec in data for rank, _ in enumerate(rec)],
    })
    return df


def save_sequence_data_dir_factory(data_dir: str):
    def save_sequence_data(data: List[List[int]], path: str):
        with open(os.path.join(data_dir, path), 'w') as f:
            lines = [
                ' '.join([str(x) for x in seq]) + '\n'
                for seq in data
            ]
            f.writelines(lines)

    return save_sequence_data


def save_pickle(obj: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def flatten(data: Iterable[List[int]]) -> Iterable:
    return (el for inner in data for el in inner)


def create_item_to_artist_map(artists: pd.DataFrame) -> dict:
    if USE_CUDF:
        return {
            item: artist for item, artist in
            zip(artists.item_id.values.get(), artists.artist_id.values.get())
        }

    return {
        item: artist for item, artist in
        zip(artists.item_id.values, artists.artist_id.values)
    }


def map_item_sequences_to_artists(sequences: Sessions, item2artist: dict) -> List[np.ndarray]:
    return [
        np.array([item2artist[item] for item in seq])
        for seq in sequences
    ]


def spm(data: Sessions, n_items: int = N_ITEMS) -> csr_matrix:
    items_count = [len(items) for items in data]
    ones = np.ones(sum(items_count))
    user_ids = np.repeat(range(len(data)), items_count)
    item_ids = list(flatten(data))
    mat = csr_matrix(
        (ones, (user_ids, item_ids)),
        shape=(len(data), n_items),
    )

    return mat


def get_seqential_user_ids(recs: Sessions) -> np.ndarray:
    return np.repeat(range(len(recs)), [len(r) for r in recs])


def create_df_from_sessions(sessions: Sessions):
    df = pd.DataFrame({
        'user_id': get_seqential_user_ids(sessions),
        'item_id': flatten(sessions),
        'rank': flatten(reversed(range(len(r))) for r in sessions)
    }, dtype='int32')
    return df


def get_group_for_lgb(sorted_vals):
    curr_val = sorted_vals[0]
    cnt = 1
    groups = []
    for val in sorted_vals[1:]:
        if val != curr_val:
            groups.append(cnt)
            curr_val = val
            cnt = 1
        else:
            cnt += 1

    groups.append(cnt)

    return np.array(groups)


def downsample_negative(df: pd.DataFrame, keep: float):
    np.random.seed(33)
    negatives = np.where((df['target'] == 0).values)[0]
    positives = np.where((df['target'] == 1).values)[0]
    num_negatives = int(np.ceil(len(negatives) * keep))
    np.random.shuffle(negatives)
    new_negatives = negatives[:num_negatives]
    index = np.concatenate([positives, new_negatives])
    index.sort()
    return df.iloc[index]


def cast(df):
    for col in df.columns:
        if df[col].dtype.name in ('Float64', 'Float32'):
            df[col] = df[col].astype(np.float32)

        if df[col].dtype.name in ('Int64', 'Int32'):
            df[col] = df[col].fillna(-1).astype(np.int64)


def polars_left_merge(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    common_columns = list(set(df1.columns) & set(df2.columns))
    return df1.join(df2, on=common_columns, how='left')


def polars_outer_merge(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    common_columns = list(set(df1.columns) & set(df2.columns))
    return df1.join(df2, on=common_columns, how='outer')


def cast_to_float32(pl_frame: pl.DataFrame):
    float_columns = [
        col for col in pl_frame.columns
        if pl_frame[col].dtype == pl.Float64 or (pl_frame[col].dtype == pl.Int64 and pl_frame[col].null_count() > 0)
    ]
    return pl_frame.with_columns([pl.col(col).cast(pl.Float32) for col in float_columns])




def read_parquet(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    for col in ['user_id', 'item_id', 'artist_id']:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int32, strict=False))

    if '__index_level_0__' in df.columns:
        df = df.drop(['__index_level_0__'])
    return df



@jit(nopython=True)
def mrr(preds, targets, groups):
    total = 0
    group_start = 0

    for group_size in groups:
        ranks = np.argsort(preds[group_start:group_start + group_size])[::-1]
        for i in range(min(len(ranks), 100)):
            if targets[group_start + ranks[i]] == 1:
                total += 1 / (i + 1)

        group_start = group_start + group_size

    return total / len(groups)


@njit(parallel=True)
def pmrr(preds, targets, groups):
    total = 0
    group_starts = np.cumsum(groups)

    for group_id in prange(len(groups)):
        group_end = group_starts[group_id]
        group_start = group_end - groups[group_id]
        ranks = np.argsort(preds[group_start: group_end])[::-1]
        for i in range(min(len(ranks), 100)):
            if targets[group_start + ranks[i]] == 1:
                total += 1 / (i + 1)

    return total / len(groups)


def lgb_mrr_wrapper(preds, lgb_dataset):
    metric = pmrr(preds, lgb_dataset.label, lgb_dataset.group)
    return 'mrr', metric, True
