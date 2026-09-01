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
_PAIR_DTYPE = np.dtype([('u', np.int64), ('i', np.int64)])


def _pair_history(users, items, times, stable_order):
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    stable_order = np.asarray(stable_order, dtype=np.int64)
    n = users.size

    if n == 0:
        return (np.zeros(0, dtype=np.int64),
                np.empty(0, dtype=_PAIR_DTYPE),
                np.zeros(0, dtype=np.int64))

    order = np.lexsort((stable_order, times, items, users))
    su = users[order]
    si = items[order]
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = (su[1:] != su[:-1]) | (si[1:] != si[:-1])
    starts = np.flatnonzero(boundary)
    lengths = np.diff(np.append(starts, n)).astype(np.int64, copy=False)

    prior_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    prior = np.empty(n, dtype=np.int64)
    prior[order] = prior_sorted

    keys = np.empty(starts.size, dtype=_PAIR_DTYPE)
    keys['u'] = su[starts]
    keys['i'] = si[starts]
    return prior, keys, lengths


def _user_history(users, times, stable_order):
    users = np.asarray(users, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    stable_order = np.asarray(stable_order, dtype=np.int64)
    n = users.size

    if n == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))

    order = np.lexsort((stable_order, times, users))
    su = users[order]
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(boundary)
    lengths = np.diff(np.append(starts, n)).astype(np.int64, copy=False)

    prior_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    prior = np.empty(n, dtype=np.int64)
    prior[order] = prior_sorted
    return prior, su[starts], lengths


def _lookup_pairs(users, items, keys, counts):
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    query = np.empty(users.size, dtype=_PAIR_DTYPE)
    query['u'] = users
    query['i'] = items
    out = np.zeros(users.size, dtype=np.int64)
    if keys.size == 0:
        return out

    pos = np.searchsorted(keys, query)
    in_range = pos < keys.size
    safe_pos = np.minimum(pos, keys.size - 1)
    matched = in_range & (keys[safe_pos] == query)
    out[matched] = counts[pos[matched]]
    return out


def _lookup_users(users, known_users, counts):
    users = np.asarray(users, dtype=np.int64)
    out = np.zeros(users.size, dtype=np.int64)
    if known_users.size == 0:
        return out

    pos = np.searchsorted(known_users, users)
    in_range = pos < known_users.size
    safe_pos = np.minimum(pos, known_users.size - 1)
    matched = in_range & (known_users[safe_pos] == users)
    out[matched] = counts[pos[matched]]
    return out


def build_features(splits, train_idx):
    Xs, feature_count = SO.build_features(splits, train_idx, OP_CONFIG)

    tr = splits['train']
    idx = np.asarray(train_idx, dtype=np.int64)
    if Xs['train'].shape[0] != tr.n:
        raise AssertionError('SO training features must retain full train row indexing')

    users = np.asarray(tr.user_id, dtype=np.int64)[idx]
    authors = np.asarray(tr.author_id, dtype=np.int64)[idx]
    videos = np.asarray(tr.video_id, dtype=np.int64)[idx]
    times = np.asarray(tr.time_ms, dtype=np.int64)[idx]

    ua_prior, ua_keys, ua_final = _pair_history(users, authors, times, idx)
    uv_prior, uv_keys, uv_final = _pair_history(users, videos, times, idx)
    user_prior, known_users, user_final = _user_history(users, times, idx)

    extras = {}
    train_extra = np.zeros((tr.n, 3), dtype=np.float32)
    selected_extra = np.empty((idx.size, 3), dtype=np.float32)
    selected_extra[:, 0] = np.log1p(ua_prior).astype(np.float32)
    selected_extra[:, 1] = np.log1p(uv_prior).astype(np.float32)
    selected_extra[:, 2] = (ua_prior.astype(np.float32) /
                            (user_prior.astype(np.float32) + np.float32(1.0)))
    train_extra[idx] = selected_extra
    extras['train'] = train_extra

    for split in Xs:
        if split == 'train':
            continue
        rows = splits[split]
        split_users = np.asarray(rows.user_id, dtype=np.int64)
        author_counts = _lookup_pairs(
            split_users, rows.author_id, ua_keys, ua_final)
        video_counts = _lookup_pairs(
            split_users, rows.video_id, uv_keys, uv_final)
        total_counts = _lookup_users(split_users, known_users, user_final)

        extra = np.empty((rows.n, 3), dtype=np.float32)
        extra[:, 0] = np.log1p(author_counts).astype(np.float32)
        extra[:, 1] = np.log1p(video_counts).astype(np.float32)
        extra[:, 2] = (author_counts.astype(np.float32) /
                       (total_counts.astype(np.float32) + np.float32(1.0)))
        extras[split] = extra

    for split in Xs:
        if Xs[split].shape[0] != extras[split].shape[0]:
            raise AssertionError('base and affinity feature row counts differ')
        Xs[split] = np.ascontiguousarray(
            np.concatenate((Xs[split], extras[split]), axis=1),
            dtype=np.float32)

    return Xs, feature_count + 3
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
