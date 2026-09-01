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
    train = splits['train']
    y_long = np.asarray(train.label, dtype=np.float32)
    aux = DV.train_targets(['is_click', 'is_profile_enter', 'is_like'])
    y_click = np.asarray(aux['is_click'], dtype=np.float32)
    y_profile = np.asarray(aux['is_profile_enter'], dtype=np.float32)
    y_like = np.asarray(aux['is_like'], dtype=np.float32)

    y_soft = 0.70 * y_long + 0.22 * y_click + 0.06 * y_profile + 0.02 * y_like
    y_soft = np.clip(y_soft, 0.0, 1.0).astype(np.float32, copy=False)

    ytr = y_soft[train_idx]
    if not np.isfinite(ytr).all():
        raise ValueError('soft_long_view: target contains NaN/Inf')
    if np.min(ytr) < 0.0 or np.max(ytr) > 1.0:
        raise ValueError('soft_long_view: target must lie in [0,1]')
    return y_soft
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


def _fm_init(dim, k, seed):
    rng = np.random.default_rng(seed)
    return {
        'w0': 0.0,
        'w': np.zeros(dim, dtype=np.float32),
        'V': rng.normal(0.0, 0.01, size=(dim, k)).astype(np.float32),
        'k': int(k),
    }


def _fm_predict_logits(model, X):
    X = np.asarray(X, dtype=np.int64)
    w = model['w']
    V = model['V']
    linear = model['w0'] + np.sum(w[X], axis=1)
    vx = V[X]
    s1 = np.sum(vx, axis=1)
    s2 = np.sum(vx * vx, axis=1)
    inter = 0.5 * np.sum(s1 * s1 - s2, axis=1)
    return (linear + inter).astype(np.float32, copy=False)


def _fm_sgd_step(model, X, grad_out, lr, l2):
    X = np.asarray(X, dtype=np.int64)
    grad_out = np.asarray(grad_out, dtype=np.float32)
    w = model['w']
    V = model['V']

    model['w0'] -= lr * float(np.sum(grad_out))

    vals, counts = np.unique(X.reshape(-1), return_counts=True)
    count_map = counts.astype(np.float32)
    grad_w = np.bincount(X.reshape(-1), weights=np.repeat(grad_out, X.shape[1]), minlength=w.shape[0]).astype(np.float32)
    w[:len(grad_w)] -= lr * (grad_w + l2 * w[:len(grad_w)] * 0.0)
    w[vals] -= lr * (l2 * w[vals])

    sx = np.sum(V[X], axis=1)
    flat_x = X.reshape(-1)
    row_rep = np.repeat(np.arange(X.shape[0]), X.shape[1])
    grad_rep = np.repeat(grad_out, X.shape[1])[:, None]
    contrib = grad_rep * (sx[row_rep] - V[flat_x])

    for f in range(V.shape[1]):
        gv = np.bincount(flat_x, weights=contrib[:, f], minlength=V.shape[0]).astype(np.float32)
        V[:, f] -= lr * gv
    V[vals] -= lr * (l2 * V[vals])


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    rng = np.random.default_rng(seed)
    Xtr = np.asarray(Xs['train'][train_idx], dtype=np.int64)
    ytr = np.asarray(y[train_idx], dtype=np.float32)
    if not np.isfinite(ytr).all():
        raise ValueError('soft_long_view_train: target contains NaN/Inf')
    if np.min(ytr) < 0.0 or np.max(ytr) > 1.0:
        raise ValueError('soft_long_view_train: target must lie in [0,1]')

    model = _fm_init(dim, int(HP['k']), seed)
    batch_size = int(HP['batch'])
    lr = float(HP['lr'])
    l2 = float(HP['l2'])
    epochs = int(HP['epochs'])

    losses = []
    n = len(train_idx)
    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]

            logits = _fm_predict_logits(model, xb)
            pred = sigmoid(logits)
            pred = np.clip(pred, 1e-8, 1.0 - 1e-8)
            err = pred - yb

            batch_loss = float(np.mean(-(yb * np.log(pred) + (1.0 - yb) * np.log(1.0 - pred))))
            epoch_loss += batch_loss * len(idx)
            seen += len(idx)

            grad_out = err / max(1, len(idx))
            _fm_sgd_step(model, xb, grad_out, lr=lr, l2=l2)

        losses.append(epoch_loss / max(1, seen))
        if verbose:
            print(json.dumps({'epoch': epoch + 1, 'train_loss': losses[-1]}))

    info = {
        'objective': 'pointwise_soft_label_bce',
        'epochs': epochs,
        'batch': batch_size,
        'lr': lr,
        'l2': l2,
        'k': int(HP['k']),
        'soft_target_weights': {
            'long_view': 0.70,
            'is_click': 0.22,
            'is_profile_enter': 0.06,
            'is_like': 0.02
        },
        'train_loss_last': float(losses[-1]) if losses else None
    }
    return model, info
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    x = np.asarray(Xs[split], dtype=np.int64)
    logits = _fm_predict_logits(model, x)
    return sigmoid(logits).astype(np.float32, copy=False)
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
