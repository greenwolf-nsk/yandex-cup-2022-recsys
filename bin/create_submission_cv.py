import argparse

import numpy as np
import tqdm
import pandas as pd
import lightgbm as lgb


parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--boosters', nargs='+', type=str)
parser.add_argument('--out')


if __name__ == '__main__':
    args = parser.parse_args()
    boosters = [
        lgb.Booster(model_file=booster)
        for booster in args.boosters
    ]

    args = parser.parse_args()
    user_recs = []
    for chunk_id in tqdm.tqdm(range(6)):
        chunk_name = f'{chunk_id}_{args.candidates}'
        test_candidates = pd.read_parquet(chunk_name)
        vals = test_candidates[boosters[0].feature_name()].to_numpy(dtype=np.float32)
        prediction_df = test_candidates[['user_id', 'item_id']]
        del test_candidates

        preds = np.zeros(len(prediction_df))
        for model in tqdm.tqdm(boosters):
            preds += model.predict(vals)

        prediction_df['pred'] = preds
        df = prediction_df.sort_values(['user_id', 'pred'], ascending=[True, False])

        for user_id, recs in tqdm.tqdm(df.groupby('user_id', sort=False)['item_id']):
            user_recs.append(recs.values.tolist()[:100])

    with open(args.out, 'w') as f:
        result = [
            ' '.join(map(str, row)) + '\n'
            for row in user_recs
        ]
        f.writelines(result)
