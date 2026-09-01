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
OP_CONFIG = {"dur_buckets": 10, "features": [], "hp": {"aux_weight": 0.2, "batch": 8192, "epochs": 12, "hidden": [128, 64], "k": 16, "l2": 0.0001, "lr": 0.001, "n_neg": 1, "torch_threads": 1}, "model_family": "torch_deepfm_mtl", "objective": "pointwise_engagement_mtl"}


def build_data_view():
    return SO.build_data_view()
# <<<END:data_view>>>


# <<<BLOCK:features>>>
def build_features(splits, train_idx):
    return SO.build_features(splits, train_idx, OP_CONFIG)
# <<<END:features>>>


# <<<BLOCK:target>>>
def build_target(splits, train_idx):
    target = dict(SO.build_target(splits, train_idx, OP_CONFIG))

    ratio, ratio_valid = DV.watch_ratio()
    ratio = np.asarray(ratio, dtype=np.float64)
    ratio_valid = np.asarray(ratio_valid, dtype=bool)
    duration_ms = np.asarray(splits['train'].duration_ms, dtype=np.float64)

    if ratio.shape[0] != splits['train'].n or ratio_valid.shape[0] != splits['train'].n:
        raise ValueError('watch_ratio arrays must cover the full training split')

    played_ms = ratio * duration_ms
    watch_mask = (ratio_valid & np.isfinite(ratio) & np.isfinite(played_ms) &
                  np.isfinite(duration_ms) & (duration_ms > 0.0) &
                  (played_ms >= 0.0))
    watch_completed = watch_mask & (played_ms >= duration_ms)
    watch_incomplete = watch_mask & ~watch_completed

    watch_target = np.zeros(splits['train'].n, dtype=np.float32)
    watch_bound = np.zeros(splits['train'].n, dtype=np.float32)
    watch_target[watch_incomplete] = (
        np.log1p(played_ms[watch_incomplete]) / 12.0
    ).astype(np.float32)
    watch_bound[watch_completed] = (
        np.log1p(duration_ms[watch_completed]) / 12.0
    ).astype(np.float32)

    target['watch_target'] = watch_target
    target['watch_bound'] = watch_bound
    target['watch_mask'] = watch_mask
    target['watch_completed'] = watch_completed
    target['watch_ratio_valid'] = ratio_valid
    target['watch_zero_duration'] = duration_ms <= 0.0
    return target
# <<<END:target>>>


# <<<BLOCK:model>>>
import torch
import torch.nn as nn
import torch.nn.functional as F

sigmoid = SO.sigmoid


class FM(nn.Module):
    def __init__(self, dimension, field_count, k=16, hidden=(128, 64), n_aux=4):
        super().__init__()
        self.dimension = int(dimension)
        self.field_count = int(field_count)
        self.k = int(k)

        self.embedding = nn.Embedding(self.dimension, self.k)
        self.linear_embedding = nn.Embedding(self.dimension, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)

        layers = []
        width = self.field_count * self.k
        for next_width in hidden:
            layers.append(nn.Linear(width, int(next_width)))
            layers.append(nn.ReLU())
            width = int(next_width)
        self.deep = nn.Sequential(*layers)

        self.main_head = nn.Linear(width, 1)
        self.engagement_head = nn.Linear(width, int(n_aux))
        self.watch_head = nn.Linear(width, 1)

    def forward_all(self, x):
        embedded = self.embedding(x)
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(-1)
        summed = embedded.sum(dim=1)
        second_order = 0.5 * (
            summed.square() - embedded.square().sum(dim=1)
        ).sum(dim=1)
        shared = self.deep(embedded.reshape(embedded.shape[0], -1))
        main = first_order + second_order + self.main_head(shared).squeeze(-1)
        engagement = self.engagement_head(shared)
        watch = self.watch_head(shared).squeeze(-1)
        return main, engagement, watch

    def forward(self, x):
        main, engagement, _ = self.forward_all(x)
        return main, engagement

    @torch.no_grad()
    def predict(self, x, batch_size=8192):
        x = np.asarray(x, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != self.field_count:
            raise ValueError('prediction features have incompatible shape')
        self.eval()
        device = next(self.parameters()).device
        outputs = []
        for start in range(0, x.shape[0], int(batch_size)):
            xb = torch.as_tensor(
                x[start:start + int(batch_size)], dtype=torch.long, device=device
            )
            main_logit, _, _ = self.forward_all(xb)
            outputs.append(torch.sigmoid(main_logit).cpu().numpy().astype(np.float32))
        if not outputs:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(outputs)
# <<<END:model>>>


# <<<BLOCK:loss>>>
loss_and_step = SO.loss_and_step
# <<<END:loss>>>


# <<<BLOCK:train>>>
HP = dict(OP_CONFIG['hp'])


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    torch.set_num_threads(int(HP['torch_threads']))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_full = int(splits['train'].n)

    def selected(values, dtype=None):
        values = np.asarray(values, dtype=dtype)
        if values.shape[0] == n_full:
            return values[train_idx]
        if values.shape[0] == train_idx.shape[0]:
            return values
        raise ValueError('training target has incompatible row count')

    x_train = np.asarray(Xs['train'][train_idx], dtype=np.int64)
    main_target = selected(y['main'], np.float32).reshape(-1)
    engagement_target = selected(y['engagement_aux'], np.float32)
    if engagement_target.ndim == 1:
        engagement_target = engagement_target.reshape(-1, 1)

    watch_target = selected(y['watch_target'], np.float32).reshape(-1)
    watch_bound = selected(y['watch_bound'], np.float32).reshape(-1)
    watch_mask = selected(y['watch_mask'], bool).reshape(-1)
    watch_completed = selected(y['watch_completed'], bool).reshape(-1)
    ratio_valid = selected(y['watch_ratio_valid'], bool).reshape(-1)
    zero_duration = selected(y['watch_zero_duration'], bool).reshape(-1)

    if not np.isfinite(main_target).all():
        raise ValueError('main target contains non-finite values')
    if np.any((main_target < 0.0) | (main_target > 1.0)):
        raise ValueError('main target must be binary')

    watch_incomplete = watch_mask & ~watch_completed
    invalid_included = watch_mask & (~ratio_valid | zero_duration)
    if np.any(invalid_included):
        raise AssertionError('invalid watch-ratio or zero-duration rows entered watch loss')

    n_neg = int(HP['n_neg'])
    if n_neg < 0:
        raise ValueError('n_neg must be nonnegative')

    rng = np.random.default_rng(int(seed))
    positive_pool = np.flatnonzero(main_target > 0.5).astype(np.int64)
    negative_pool = np.flatnonzero(main_target <= 0.5).astype(np.int64)

    if positive_pool.size and negative_pool.size and n_neg > 0:
        requested_negatives = int(positive_pool.size) * n_neg
        sampled_negative_count = min(int(negative_pool.size), requested_negatives)
        sampled_negatives = rng.choice(
            negative_pool, size=sampled_negative_count, replace=False
        ).astype(np.int64, copy=False)
        sampled_rows = np.concatenate((positive_pool, sampled_negatives))
    elif positive_pool.size:
        sampled_negatives = np.empty(0, dtype=np.int64)
        sampled_rows = positive_pool.copy()
    else:
        sampled_negatives = negative_pool.copy()
        sampled_rows = negative_pool.copy()

    if sampled_rows.size == 0:
        raise ValueError('negative sampler produced no training rows')

    model = FM(
        dimension=int(dim),
        field_count=int(x_train.shape[1]),
        k=int(HP['k']),
        hidden=tuple(HP['hidden']),
        n_aux=int(engagement_target.shape[1]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(HP['lr']), weight_decay=float(HP['l2'])
    )

    batch_size = int(HP['batch'])
    epochs = int(HP['epochs'])
    engagement_weight = float(HP['aux_weight'])
    watch_weight = 0.2
    final_main_loss = 0.0
    final_engagement_loss = 0.0
    final_watch_loss = 0.0

    model.train()
    for epoch in range(epochs):
        order = sampled_rows[rng.permutation(sampled_rows.shape[0])]
        main_sum = 0.0
        engagement_sum = 0.0
        watch_sum = 0.0
        row_count = 0
        watch_count = 0

        for start in range(0, order.shape[0], batch_size):
            batch = order[start:start + batch_size]
            xb = torch.from_numpy(x_train[batch]).long()
            main_yb = torch.from_numpy(main_target[batch]).float()
            engagement_yb = torch.from_numpy(engagement_target[batch]).float()
            watch_yb = torch.from_numpy(watch_target[batch]).float()
            bound_yb = torch.from_numpy(watch_bound[batch]).float()
            mask_b = torch.from_numpy(watch_mask[batch]).bool()
            completed_b = torch.from_numpy(watch_completed[batch]).bool()

            main_logit, engagement_logit, watch_prediction = model.forward_all(xb)
            main_loss = F.binary_cross_entropy_with_logits(main_logit, main_yb)
            engagement_loss = F.binary_cross_entropy_with_logits(
                engagement_logit, engagement_yb
            )

            incomplete_b = mask_b & ~completed_b
            watch_numerator = watch_prediction.new_zeros(())
            if bool(incomplete_b.any()):
                residual = watch_prediction[incomplete_b] - watch_yb[incomplete_b]
                watch_numerator = watch_numerator + residual.square().sum()
            if bool(completed_b.any()):
                shortfall = F.relu(bound_yb[completed_b] - watch_prediction[completed_b])
                watch_numerator = watch_numerator + shortfall.square().sum()

            valid_count = int(mask_b.sum().item())
            if valid_count:
                watch_loss = watch_numerator / float(valid_count)
            else:
                watch_loss = watch_prediction.sum() * 0.0

            total_loss = (main_loss + engagement_weight * engagement_loss +
                          watch_weight * watch_loss)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            batch_rows = int(batch.shape[0])
            main_sum += float(main_loss.detach()) * batch_rows
            engagement_sum += float(engagement_loss.detach()) * batch_rows
            watch_sum += float(watch_loss.detach()) * valid_count
            row_count += batch_rows
            watch_count += valid_count

        final_main_loss = main_sum / max(row_count, 1)
        final_engagement_loss = engagement_sum / max(row_count, 1)
        final_watch_loss = watch_sum / max(watch_count, 1)
        if verbose:
            print(json.dumps({
                'epoch': epoch + 1,
                'main_loss': final_main_loss,
                'engagement_loss': final_engagement_loss,
                'watch_loss': final_watch_loss,
                'sampled_rows': int(sampled_rows.size),
            }))

    if not np.isfinite(final_watch_loss) or final_watch_loss <= 0.0:
        raise AssertionError('watch-head loss must be finite and nonzero')

    consumed_watch_mask = watch_mask[sampled_rows]
    consumed_completed = watch_completed[sampled_rows]
    consumed_invalid = invalid_included[sampled_rows]
    consumed_zero_duration = zero_duration[sampled_rows]

    model.eval()
    info = {
        'epochs': epochs,
        'checkpoint_policy': 'fixed_final_epoch',
        'main_loss': float(final_main_loss),
        'engagement_loss': float(final_engagement_loss),
        'watch_loss': float(final_watch_loss),
        'watch_weight': float(watch_weight),
        'n_neg': int(n_neg),
        'sampler': 'all_positive_fixed_negative_subsample',
        'sampler_positive_pool_count': int(positive_pool.size),
        'sampler_negative_pool_count': int(negative_pool.size),
        'sampler_sampled_negative_count': int(sampled_negatives.size),
        'sampler_training_row_count': int(sampled_rows.size),
        'watch_incomplete_count': int((consumed_watch_mask & ~consumed_completed).sum()),
        'watch_completed_count': int((consumed_watch_mask & consumed_completed).sum()),
        'watch_valid_count': int(consumed_watch_mask.sum()),
        'watch_invalid_or_zero_included': int(consumed_invalid.sum()),
        'watch_zero_duration_excluded': int(consumed_zero_duration.sum()),
    }
    return model, info
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
