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
    labels = np.asarray(splits['train'].label)[train_idx].astype(np.float32, copy=False)
    users = np.asarray(splits['train'].user_id)[train_idx]
    DV.assert_trainable(labels, 'build_target.long_view')
    if labels.ndim != 1 or users.ndim != 1 or len(labels) != len(users):
        raise ValueError('training labels and user IDs must be aligned one-dimensional arrays')
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError('long_view must be binary')
    return {'long_view': labels, 'user_id': users}
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

import torch
import torch.nn.functional as F


class _UniformUserFM(torch.nn.Module):
    def __init__(self, dimension, k):
        super().__init__()
        self.dimension = int(dimension)
        self.k = int(k)
        self.linear = torch.nn.Embedding(self.dimension, 1, sparse=False)
        self.factors = torch.nn.Embedding(self.dimension, self.k, sparse=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.factors.weight.normal_(mean=0.0, std=0.01)

    def forward(self, x):
        linear_score = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (summed.square() - v.square().sum(dim=1)).sum(dim=1)
        return self.bias[0] + linear_score + interaction


def _build_pair_pools(labels, user_inverse, n_users):
    positive_mask = labels == 1.0
    negative_mask = labels == 0.0
    positive_counts = np.bincount(user_inverse[positive_mask], minlength=n_users)
    negative_counts = np.bincount(user_inverse[negative_mask], minlength=n_users)
    eligible_users = np.flatnonzero((positive_counts > 0) & (negative_counts > 0))

    positive_rows = np.flatnonzero(positive_mask)
    negative_rows = np.flatnonzero(negative_mask)
    if len(positive_rows):
        positive_rows = positive_rows[np.argsort(user_inverse[positive_rows], kind='stable')]
    if len(negative_rows):
        negative_rows = negative_rows[np.argsort(user_inverse[negative_rows], kind='stable')]

    positive_starts = np.cumsum(np.r_[0, positive_counts[:-1]], dtype=np.int64)
    negative_starts = np.cumsum(np.r_[0, negative_counts[:-1]], dtype=np.int64)
    return (positive_counts, negative_counts, eligible_users, positive_rows,
            negative_rows, positive_starts, negative_starts)


def _sample_pair_indices(rng, pair_count, user_inverse, pools, mode):
    (positive_counts, negative_counts, eligible_users, positive_rows,
     negative_rows, positive_starts, negative_starts) = pools
    if pair_count <= 0 or len(eligible_users) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    if mode == 'uniform_user':
        choices = rng.integers(0, len(eligible_users), size=pair_count)
        sampled_users = eligible_users[choices]
        pos_random = rng.random(pair_count)
        pos_offsets = np.floor(pos_random * positive_counts[sampled_users]).astype(np.int64)
        pos_index = positive_rows[positive_starts[sampled_users] + pos_offsets]
    elif mode == 'row_weighted':
        eligible_flag = np.zeros(len(positive_counts), dtype=bool)
        eligible_flag[eligible_users] = True
        eligible_positive_rows = positive_rows[eligible_flag[user_inverse[positive_rows]]]
        choices = rng.integers(0, len(eligible_positive_rows), size=pair_count)
        pos_index = eligible_positive_rows[choices]
        sampled_users = user_inverse[pos_index]
        rng.random(pair_count)
    else:
        raise ValueError('unknown pair sampling mode: %s' % mode)

    neg_random = rng.random(pair_count)
    neg_offsets = np.floor(neg_random * negative_counts[sampled_users]).astype(np.int64)
    neg_index = negative_rows[negative_starts[sampled_users] + neg_offsets]
    return pos_index, neg_index, sampled_users


def _bpr_loss(z_pos, z_neg):
    if z_pos.ndim != 1 or z_neg.ndim != 1 or z_pos.shape != z_neg.shape:
        raise ValueError('BPR score vectors must be aligned and one-dimensional')
    if z_pos.numel() == 0:
        return None
    return F.softplus(-(z_pos - z_neg)).sum() / z_pos.numel()


def _run_training_fixtures():
    z_pos = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    z_neg = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    sign_loss = _bpr_loss(z_pos, z_neg)
    sign_loss.backward()
    g_pos = float(z_pos.grad[0])
    g_neg = float(z_neg.grad[0])
    updated_margin = (-g_pos) - (-g_neg)
    if not (g_pos < 0.0 and g_neg > 0.0 and updated_margin > 0.0):
        raise AssertionError('BPR sign fixture failed')

    fixture_users = np.repeat(np.arange(4, dtype=np.int64), 2)
    fixture_labels = np.tile(np.array([1.0, 0.0], dtype=np.float32), 4)
    fixture_pools = _build_pair_pools(fixture_labels, fixture_users, 4)
    uniform_pairs = _sample_pair_indices(
        np.random.default_rng(173), 64, fixture_users, fixture_pools, 'uniform_user')
    row_pairs = _sample_pair_indices(
        np.random.default_rng(173), 64, fixture_users, fixture_pools, 'row_weighted')
    if not (np.array_equal(uniform_pairs[0], row_pairs[0]) and
            np.array_equal(uniform_pairs[1], row_pairs[1]) and
            np.array_equal(uniform_pairs[2], row_pairs[2])):
        raise AssertionError('equal-positive-mass sampler parity fixture failed')

    fixture_scores = np.array([0.7, -0.2, 0.1, 0.4, -0.3, -0.8, 0.9, 0.0],
                              dtype=np.float64)
    up = torch.tensor(fixture_scores[uniform_pairs[0]], requires_grad=True)
    un = torch.tensor(fixture_scores[uniform_pairs[1]], requires_grad=True)
    rp = torch.tensor(fixture_scores[row_pairs[0]], requires_grad=True)
    rn = torch.tensor(fixture_scores[row_pairs[1]], requires_grad=True)
    uniform_loss = _bpr_loss(up, un)
    row_loss = _bpr_loss(rp, rn)
    uniform_loss.backward()
    row_loss.backward()
    if not (torch.allclose(uniform_loss, row_loss, rtol=0.0, atol=1e-12) and
            torch.allclose(up.grad, rp.grad, rtol=0.0, atol=1e-12) and
            torch.allclose(un.grad, rn.grad, rtol=0.0, atol=1e-12)):
        raise AssertionError('row-weighted BPR loss/gradient parity fixture failed')


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    torch.set_num_threads(1)
    _run_training_fixtures()

    labels = np.asarray(y['long_view'], dtype=np.float32)
    user_ids = np.asarray(y['user_id'])
    x_train = np.asarray(Xs['train'][train_idx])
    if labels.ndim != 1 or user_ids.ndim != 1 or x_train.ndim != 2:
        raise ValueError('training inputs must have the expected dimensions')
    if len(x_train) != len(labels) or len(user_ids) != len(labels):
        raise ValueError('training feature, target, and user rows are not aligned')
    if x_train.size and (x_train.min() < 0 or x_train.max() >= int(dim)):
        raise ValueError('globally offset feature ID is outside the FM table')

    _, user_inverse = np.unique(user_ids, return_inverse=True)
    user_inverse = user_inverse.astype(np.int64, copy=False)
    n_users = int(user_inverse.max()) + 1 if len(user_inverse) else 0
    pools = _build_pair_pools(labels, user_inverse, n_users)
    positive_counts, negative_counts, eligible_users = pools[:3]

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = _UniformUserFM(dim, int(HP['k']))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(HP['lr']), weight_decay=float(HP['l2']))

    epochs = int(HP['epochs'])
    batch_size = int(HP['batch'])
    n_neg = int(HP['n_neg'])
    pairs_per_epoch = int((labels == 1.0).sum()) * n_neg
    first_epoch_user_counts = None
    total_pairs = 0
    last_loss = None

    model.train()
    if len(eligible_users) and pairs_per_epoch > 0:
        eligible_lookup = np.full(n_users, -1, dtype=np.int64)
        eligible_lookup[eligible_users] = np.arange(len(eligible_users), dtype=np.int64)
        for epoch in range(epochs):
            epoch_user_counts = np.zeros(len(eligible_users), dtype=np.int64) if epoch == 0 else None
            epoch_loss_sum = 0.0
            epoch_pairs = 0
            for offset in range(0, pairs_per_epoch, batch_size):
                requested = min(batch_size, pairs_per_epoch - offset)
                pos_index, neg_index, sampled_users = _sample_pair_indices(
                    rng, requested, user_inverse, pools, 'uniform_user')
                valid_pairs = len(pos_index)
                if valid_pairs == 0:
                    continue
                if epoch_user_counts is not None:
                    sampled_slots = eligible_lookup[sampled_users]
                    epoch_user_counts += np.bincount(
                        sampled_slots, minlength=len(eligible_users))

                pair_features = np.concatenate(
                    (x_train[pos_index], x_train[neg_index]), axis=0)
                pair_tensor = torch.as_tensor(pair_features, dtype=torch.long)
                scores = model(pair_tensor)
                z_pos = scores[:valid_pairs]
                z_neg = scores[valid_pairs:]
                loss = _bpr_loss(z_pos, z_neg)
                if loss is None:
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach()) * valid_pairs
                epoch_pairs += valid_pairs
                total_pairs += valid_pairs

            if epoch_user_counts is not None:
                first_epoch_user_counts = epoch_user_counts
            last_loss = epoch_loss_sum / max(epoch_pairs, 1)
            if verbose:
                print('epoch=%d pairs=%d loss=%.6f' %
                      (epoch + 1, epoch_pairs, last_loss))

    uniform_count_cv = None
    if first_epoch_user_counts is not None and first_epoch_user_counts.mean() > 0:
        uniform_count_cv = float(
            first_epoch_user_counts.std() / first_epoch_user_counts.mean())

    model.eval()
    info = {
        'trainer': 'uniform_user_fm_bpr',
        'eligible_users': int(len(eligible_users)),
        'train_users': int(n_users),
        'eligible_positive_rows': int(positive_counts[eligible_users].sum()) if len(eligible_users) else 0,
        'eligible_negative_rows': int(negative_counts[eligible_users].sum()) if len(eligible_users) else 0,
        'pairs_per_epoch': int(pairs_per_epoch),
        'total_pairs': int(total_pairs),
        'epochs': int(epochs),
        'n_neg': int(n_neg),
        'last_train_loss': None if last_loss is None else float(last_loss),
        'first_epoch_user_count_cv': uniform_count_cv,
        'objective': 'mean_softplus_negative_pair_margin',
        'l2': float(HP['l2']),
        'l2_implementation': 'global_adam_weight_decay',
        'checkpoint_policy': 'fixed_final_epoch'
    }
    return model, info
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    x = np.asarray(Xs[split])
    if x.ndim != 2:
        raise ValueError('FM inference features must be two-dimensional')
    output = np.empty(len(x), dtype=np.float32)
    batch_size = 65536
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:stop], dtype=torch.long)
            output[start:stop] = model(xb).cpu().numpy().astype(np.float32, copy=False)
    return output
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
