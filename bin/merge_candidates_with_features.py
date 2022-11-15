import argparse

import polars as pl

from lib.utils import read_parquet, polars_left_merge, read_data, create_targets_df, cast_to_float32

parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--artists')
parser.add_argument('--features', nargs='+')
parser.add_argument('--targets')
parser.add_argument('--mode')
parser.add_argument('--create_composite_features', type=bool)
parser.add_argument('--out')


if __name__ == '__main__':
    args = parser.parse_args()
    candidates = cast_to_float32(read_parquet(args.candidates))
    artists_df = read_parquet(args.artists)
    candidates = polars_left_merge(candidates, artists_df)
    features_paths = args.features


    for feature_df_path in features_paths:
        df = cast_to_float32(read_parquet(feature_df_path))
        print(f'merging {feature_df_path}')
        candidates = polars_left_merge(candidates, df)

    if args.mode == 'val' and args.targets:
        targets = create_targets_df(read_data(args.targets)).with_columns(
            [pl.col('user_id').cast(pl.Int32), pl.col('item_id').cast(pl.Int32)]
        )
        candidates = polars_left_merge(candidates, targets)
        candidates = candidates.with_columns(pl.col('target').fill_null(0))

    if args.create_composite_features:
        candidates = candidates.with_columns((pl.col('artist_like_count') / pl.col('total_likes')).alias('artist_like_share'))

    cast_to_float32(candidates).sort([pl.col("user_id"), pl.col("item_id")]).write_parquet(args.out, use_pyarrow=True)






