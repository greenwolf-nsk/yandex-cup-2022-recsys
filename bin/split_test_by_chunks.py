import argparse

import tqdm
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--candidates')


if __name__ == '__main__':
    args = parser.parse_args()
    test_candidates = pd.read_parquet(args.candidates)
    chunks = [
        (i, i * 50_000, ((i + 1) * 50000))
        for i in range(6)
    ]
    for chunk_id, chunk_start, chunk_end in tqdm.tqdm(chunks):
        chunk_name = f'{chunk_id}_{args.candidates}'
        test_candidates[test_candidates['user_id'].between(chunk_start, chunk_end - 1)].to_parquet(chunk_name)
        print(f'written chunk {chunk_start}-{chunk_end}')

