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
OP_CONFIG = {"dur_buckets": 10, "features": ["hour", "user_gap"], "hp": {"aux_weight": 0.1, "batch": 8192, "epochs": 3, "k": 32, "l2": 1e-06, "lr": 0.0005, "n_neg": 4}, "objective": "bpr_censored_watch"}


def build_data_view():
    return SO.build_data_view()
# <<<END:data_view>>>


# <<<BLOCK:features>>>
def build_features(splits, train_idx):
    return SO.build_features(splits, train_idx, OP_CONFIG)
# <<<END:features>>>


# <<<BLOCK:target>>>
def build_target(splits, train_idx):
    y_all = np.asarray(splits['train'].label, dtype=np.float32)
    y = y_all[np.asarray(train_idx, dtype=np.int64)]
    DV.assert_trainable(y, 'target.long_view')
    if y.ndim != 1 or not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError('long_view must be a one-dimensional binary target')
    return y
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PairwiseFM(nn.Module):
        def __init__(self, dimension, k):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))
            self.linear = nn.Embedding(dimension, 1)
            self.factors = nn.Embedding(dimension, k)
            nn.init.zeros_(self.linear.weight)
            nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        def forward(self, x):
            linear_term = self.linear(x).sum(dim=1).squeeze(-1)
            v = self.factors(x)
            summed = v.sum(dim=1)
            interaction = 0.5 * (summed.square() - v.square().sum(dim=1)).sum(dim=1)
            return self.bias + linear_term + interaction

    train_idx = np.asarray(train_idx, dtype=np.int64)
    x_train = np.asarray(Xs['train'][train_idx], dtype=np.int64)
    y_train = np.asarray(y, dtype=np.float32)
    users = np.asarray(splits['train'].user_id)[train_idx]
    if x_train.ndim != 2 or len(x_train) != len(y_train) or len(users) != len(y_train):
        raise ValueError('misaligned training arrays')
    if x_train.size and (x_train.min() < 0 or x_train.max() >= int(dim)):
        raise ValueError('categorical feature id outside embedding dimension')

    pos_rows = np.flatnonzero(y_train > 0.5)
    neg_rows = np.flatnonzero(y_train <= 0.5)
    pos_order = np.argsort(users[pos_rows], kind='stable')
    neg_order = np.argsort(users[neg_rows], kind='stable')
    pos_rows = pos_rows[pos_order]
    neg_rows = neg_rows[neg_order]
    pos_users, pos_starts, pos_counts = np.unique(
        users[pos_rows], return_index=True, return_counts=True)
    neg_users, neg_starts, neg_counts = np.unique(
        users[neg_rows], return_index=True, return_counts=True)
    eligible_users, pos_map, neg_map = np.intersect1d(
        pos_users, neg_users, assume_unique=True, return_indices=True)
    if len(eligible_users) == 0:
        raise ValueError('no users contain both long_view classes')

    pos_starts = pos_starts[pos_map].astype(np.int64, copy=False)
    neg_starts = neg_starts[neg_map].astype(np.int64, copy=False)
    pos_counts = pos_counts[pos_map].astype(np.int64, copy=False)
    neg_counts = neg_counts[neg_map].astype(np.int64, copy=False)
    pairs_per_user = 8
    pair_count = int(len(eligible_users) * pairs_per_user)

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = PairwiseFM(int(dim), int(HP['k']))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(HP['lr']), weight_decay=float(HP['l2']))
    batch_size = int(HP['batch'])
    epochs = int(HP['epochs'])
    epoch_losses = []
    cross_user_pairs = 0

    model.train()
    for epoch in range(epochs):
        shape = (len(eligible_users), pairs_per_user)
        pos_offsets = np.floor(rng.random(shape) * pos_counts[:, None]).astype(np.int64)
        neg_offsets = np.floor(rng.random(shape) * neg_counts[:, None]).astype(np.int64)
        sampled_pos = pos_rows[pos_starts[:, None] + pos_offsets].reshape(-1)
        sampled_neg = neg_rows[neg_starts[:, None] + neg_offsets].reshape(-1)

        mismatches = int(np.count_nonzero(users[sampled_pos] != users[sampled_neg]))
        cross_user_pairs += mismatches
        if mismatches:
            raise AssertionError('same-user pair sampler produced cross-user pairs')

        order = rng.permutation(pair_count)
        sampled_pos = sampled_pos[order]
        sampled_neg = sampled_neg[order]
        loss_sum = 0.0
        seen_pairs = 0
        for start in range(0, pair_count, batch_size):
            stop = min(start + batch_size, pair_count)
            p = torch.from_numpy(x_train[sampled_pos[start:stop]])
            n = torch.from_numpy(x_train[sampled_neg[start:stop]])
            optimizer.zero_grad(set_to_none=True)
            diff = model(p) - model(n)
            loss = F.softplus(-diff).mean()
            loss.backward()
            optimizer.step()
            count = stop - start
            loss_sum += float(loss.detach()) * count
            seen_pairs += count
        epoch_losses.append(loss_sum / max(seen_pairs, 1))
        if verbose:
            print('epoch=%d pairs=%d loss=%.6f' % (
                epoch + 1, seen_pairs, epoch_losses[-1]))

    model.eval()
    info = {
        'model': 'macro_user_balanced_pairwise_fm',
        'eligible_user_count': int(len(eligible_users)),
        'pairs_per_eligible_user_per_epoch': int(pairs_per_user),
        'pairs_per_epoch': int(pair_count),
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'embedding_k': int(HP['k']),
        'learning_rate': float(HP['lr']),
        'l2': float(HP['l2']),
        'zero_cross_user_pairs': int(cross_user_pairs),
        'train_loss_by_epoch': [float(v) for v in epoch_losses],
        'checkpoint_policy': 'fixed_final_epoch'
    }
    wrapped = {'net': model, 'info': info}
    return wrapped, info
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    import hashlib
    import torch

    net = model['net']
    x = np.asarray(Xs[split], dtype=np.int64)
    batch_size = int(HP['batch'])
    outputs = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size])
            outputs.append(net(xb).cpu().numpy().astype(np.float32, copy=False))
    pred = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    model['info']['prediction_hash'] = hashlib.sha256(
        np.ascontiguousarray(pred).view(np.uint8)).hexdigest()
    return pred
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
