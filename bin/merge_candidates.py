import argparse

from lib.utils import read_parquet, polars_outer_merge, cast_to_float32

parser = argparse.ArgumentParser()
parser.add_argument('candidates', nargs='+')
parser.add_argument('--out')


if __name__ == '__main__':
    args = parser.parse_args()
    paths = args.candidates
    merged = cast_to_float32(read_parquet(paths[0]))
    for path in paths[1:]:
        df = cast_to_float32(read_parquet(path))
        print(path)
        merged = polars_outer_merge(merged, df)

    merged.write_parquet(args.out, use_pyarrow=True)






