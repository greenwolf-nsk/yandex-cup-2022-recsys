from typing import List
from collections import Counter, defaultdict


def train_pairs(data: List[List[int]], max_candidates: int = 500):
    pairs = defaultdict(Counter)
    top_pairs = {}

    for record in data:
        for i in range(len(record) - 1):
            last, ans = record[i], record[i + 1]
            pairs[last][ans] += 1

    for key in pairs:
        top_pairs[key] = pairs[key].most_common(max_candidates)

    return top_pairs


def recommend_one_user(pairs: dict, user_data: List[int], num_recs: int = 100):
    top_pair_items = pairs.get(user_data[-1], [])
    us = set(user_data)

    return [item[0] for item in top_pair_items if item[0] not in us][:num_recs]


def read_data(path: str) -> List[List[int]]:
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(list(map(int, line.strip().split(' '))))

    return data


if __name__ == '__main__':
    train = read_data('data/train')
    test = read_data('data/test')
    pairs = train_pairs(train)
    with open('ans.csv', 'w') as f:
        for user in test:
            recs = recommend_one_user(pairs, user) or [1]
            f.write(' '.join([str(item_id) for item_id in recs]) + '\n')

