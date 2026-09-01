"""候选 pipeline —— agent 唯一可以修改的文件。初始内容 = 官方 FM baseline。

结构约定：源码被 `# <<<BLOCK:name>>> ... # <<<END:name>>>` 切成 7 段。
patch_scope 只能引用这些 block 名；一轮最多改 3 个，且必须声明一个 primary_block。
这让 metric delta 的归因是结构性保证，而不是靠 prompt 约束。

数据只能经 /task/dataview.py 取。禁止直接 open CSV —— G2 会在执行前用 AST 拦下。

用法（由 orchestrator 调用，不手动跑）：
    python3 pipeline.py --split test --seed 0 --out pred.npy
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/task')
import dataview as DV

# <<<BLOCK:data_view>>>
# Stable operator adapter. Configuration is controller-generated.
import stable_ops as SO
OP_CONFIG = {"dur_buckets": 10, "features": [], "hp": {"aux_weight": 0.2, "batch": 8192, "epochs": 8, "k": 16, "l2": 1e-06, "lr": 0.001, "n_neg": 1, "num_boost_round": 100}, "model_family": "lightgbm_rank", "objective": "pointwise"}


def build_data_view():
    return SO.build_data_view()
# <<<END:data_view>>>


# <<<BLOCK:features>>>
def _fit_sorted_vocab(values):
    values = np.asarray(values)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(values)


def _map_to_train_vocab(values, vocab):
    values = np.asarray(values)
    out = np.full(values.shape, vocab.size, dtype=np.int64)
    if vocab.size == 0 or values.size == 0:
        return out
    pos = np.searchsorted(vocab, values)
    valid = pos < vocab.size
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        pos_valid = pos[valid_idx]
        exact = vocab[pos_valid] == values[valid_idx]
        if exact.any():
            out[valid_idx[exact]] = pos_valid[exact]
    return out


def _fit_pair_counts(left_idx, right_idx, right_vocab_size):
    base = np.int64(right_vocab_size + 1)
    keys = left_idx.astype(np.int64, copy=False) * base + right_idx.astype(np.int64, copy=False)
    uniq, counts = np.unique(keys, return_counts=True)
    return uniq.astype(np.int64, copy=False), counts.astype(np.float32, copy=False), base


def _lookup_sorted_counts(query_keys, uniq_keys, counts):
    out = np.zeros(query_keys.shape[0], dtype=np.float32)
    if query_keys.size == 0 or uniq_keys.size == 0:
        return out
    pos = np.searchsorted(uniq_keys, query_keys)
    valid = pos < uniq_keys.size
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        pos_valid = pos[valid_idx]
        exact = uniq_keys[pos_valid] == query_keys[valid_idx]
        if exact.any():
            out[valid_idx[exact]] = counts[pos_valid[exact]]
    return out


def build_features(splits, train_idx):
    Xs, dim = SO.build_features(splits, train_idx, OP_CONFIG)

    train = splits['train']
    tr_user = np.asarray(train.user_id)[train_idx]
    tr_author = np.asarray(train.author_id)[train_idx]
    tr_video = np.asarray(train.video_id)[train_idx]
    tr_tab = np.asarray(train.tab)[train_idx]

    user_vocab = _fit_sorted_vocab(tr_user)
    author_vocab = _fit_sorted_vocab(tr_author)
    video_vocab = _fit_sorted_vocab(tr_video)
    tab_vocab = _fit_sorted_vocab(tr_tab)

    tr_user_idx = _map_to_train_vocab(tr_user, user_vocab)
    tr_author_idx = _map_to_train_vocab(tr_author, author_vocab)
    tr_video_idx = _map_to_train_vocab(tr_video, video_vocab)
    tr_tab_idx = _map_to_train_vocab(tr_tab, tab_vocab)

    user_total = np.bincount(tr_user_idx, minlength=user_vocab.size).astype(np.float32, copy=False)

    ua_keys, ua_counts, ua_base = _fit_pair_counts(tr_user_idx, tr_author_idx, author_vocab.size)
    uv_keys, uv_counts, uv_base = _fit_pair_counts(tr_user_idx, tr_video_idx, video_vocab.size)
    ut_keys, ut_counts, ut_base = _fit_pair_counts(tr_user_idx, tr_tab_idx, tab_vocab.size)

    for split, X in Xs.items():
        rows = splits[split]
        user_idx = _map_to_train_vocab(np.asarray(rows.user_id), user_vocab)
        author_idx = _map_to_train_vocab(np.asarray(rows.author_id), author_vocab)
        video_idx = _map_to_train_vocab(np.asarray(rows.video_id), video_vocab)
        tab_idx = _map_to_train_vocab(np.asarray(rows.tab), tab_vocab)

        ua = _lookup_sorted_counts(
            user_idx.astype(np.int64, copy=False) * ua_base + author_idx.astype(np.int64, copy=False),
            ua_keys,
            ua_counts,
        )
        uv = _lookup_sorted_counts(
            user_idx.astype(np.int64, copy=False) * uv_base + video_idx.astype(np.int64, copy=False),
            uv_keys,
            uv_counts,
        )
        ut = _lookup_sorted_counts(
            user_idx.astype(np.int64, copy=False) * ut_base + tab_idx.astype(np.int64, copy=False),
            ut_keys,
            ut_counts,
        )

        user_seen = np.zeros(rows.n, dtype=np.float32)
        known_user = user_idx < user_vocab.size
        if known_user.any():
            user_seen[known_user] = user_total[user_idx[known_user]]

        extra = np.column_stack([
            np.log1p(ua),
            np.log1p(uv),
            np.log1p(ut),
            ua / np.maximum(user_seen, 1.0),
        ]).astype(np.float32, copy=False)

        Xs[split] = np.concatenate([np.asarray(X, dtype=np.float32), extra], axis=1)

    return Xs, dim + 4
# <<<END:features>>>


# <<<BLOCK:target>>>
def build_target(splits, train_idx):
    return SO.build_target(splits, train_idx, OP_CONFIG)
# <<<END:target>>>


# <<<BLOCK:model>>>
sigmoid = SO.sigmoid
FM = SO.FM
# <<<END:model>>>


# <<<BLOCK:loss>>>
loss_and_step = SO.loss_and_step
# <<<END:loss>>>


# <<<BLOCK:train>>>
HP = dict(OP_CONFIG['hp'])


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    config = dict(OP_CONFIG)
    config['hp'] = dict(HP)
    return SO.train(splits, train_idx, Xs, dim, y, seed, config, verbose)
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    return SO.predict(model, Xs, split)
# <<<END:predict>>>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='pred.npy')
    ap.add_argument('--meta', default='meta.json')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--smoke', type=int, default=0,
                    help='冒烟模式：只用这么多训练行、最多 1 轮。用于在正式多种子评测前'
                         '几秒钟内暴露 NameError/形状错误，不产出可用分数。')
    a = ap.parse_args()

    t0 = time.time()
    splits, train_idx = build_data_view()
    if a.smoke:
        # 脚手架级降配：对 block 内容不做任何假设，只截训练行；若 HP 存在则顺带压 epoch。
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(train_idx, size=min(a.smoke, len(train_idx)),
                                       replace=False))
        if isinstance(globals().get('HP'), dict):
            HP['epochs'] = 1
            # Common non-NumPy model budgets. This is only a compile/runtime smoke,
            # never a scored experiment, so expensive estimators get a tiny fit.
            for key in ('num_boost_round', 'n_estimators', 'max_iter', 'steps'):
                if key in HP:
                    HP[key] = min(int(HP[key]), 5)
    Xs, dim = build_features(splits, train_idx)
    y = build_target(splits, train_idx)
    model, info = train(splits, train_idx, Xs, dim, y, a.seed, a.verbose)
    pred = predict(model, Xs, a.split)

    n_expect = splits[a.split].n
    assert np.isfinite(pred).all(), '预测含 NaN/Inf'
    np.save(a.out, pred)

    info.update(split=a.split, seed=a.seed, n_rows=int(n_expect),
                feature_dim=dim, wall_s=round(time.time() - t0, 1))
    with open(a.meta, 'w') as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
