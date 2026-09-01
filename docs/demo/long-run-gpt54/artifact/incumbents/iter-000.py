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
# 取哪些行进训练。初始：全部 14 天 train。
def build_data_view():
    splits = DV.load()
    train_idx = np.arange(splits['train'].n)
    return splits, train_idx
# <<<END:data_view>>>


# <<<BLOCK:features>>>
# 特征域定义与编码。类别值映射成连续 id，未见过的落到该域的 UNK 槽。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
N_DUR_BUCKETS = 10


def _raw_fields(rs, dur_edges):
    return [rs.user_id, rs.video_id, rs.author_id, rs.tab,
            np.searchsorted(dur_edges, rs.duration_ms).astype(np.int32)]


def build_features(splits, train_idx):
    tr = splits['train']
    dur_edges = np.quantile(tr.duration_ms[train_idx],
                            np.linspace(0, 1, N_DUR_BUCKETS + 1)[1:-1])

    # 词表只在 train 上建；UNK 槽在每个域末尾
    vocabs, field_dims = [], []
    for col in _raw_fields(tr, dur_edges):
        uniq = np.unique(col[train_idx])
        vocabs.append(uniq)
        field_dims.append(len(uniq) + 1)          # +1 = UNK
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    def encode(rs):
        X = np.empty((rs.n, len(FIELDS)), dtype=np.int32)
        for i, col in enumerate(_raw_fields(rs, dur_edges)):
            pos = np.searchsorted(vocabs[i], col)
            pos = np.clip(pos, 0, len(vocabs[i]) - 1)
            hit = vocabs[i][pos] == col
            X[:, i] = np.where(hit, pos, len(vocabs[i])) + offsets[i]
        return X

    return {name: encode(rs) for name, rs in splits.items()}, int(sum(field_dims))
# <<<END:features>>>


# <<<BLOCK:target>>>
# 训练目标。初始：官方二元标签 long_view。
# 反馈信号（play_time_ms / is_click / ...）只能经 DV.train_targets() 当**目标**，
# 不能当任何 split 的输入特征 —— 评测行根本没有这些值。
def build_target(splits, train_idx):
    y = splits['train'].label[train_idx].astype(np.float32)
    return DV.assert_trainable(y, where='target')
# <<<END:target>>>


# <<<BLOCK:model>>>
# 打分函数。初始：Factorization Machine，k=16，Adam。
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                    # (B,F,k)
        S = E.sum(1)                                     # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

    def state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def restore(self, st):
        self.V, self.W, self.b = st
# <<<END:model>>>


# <<<BLOCK:loss>>>
# 目标函数。初始：pointwise logloss。
def loss_and_step(model, X, y):
    B = len(y)
    z, E, S = model.logits(X)
    p = sigmoid(z)
    g = ((p - y) / B).astype(np.float32)
    gV = np.zeros_like(model.V); gW = np.zeros_like(model.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += model.l2 * model.V; gW += model.l2 * model.W

    model.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((model.V, gV, model.mV, model.vV),
                        (model.W, gW, model.mW, model.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= model.lr * (M / (1 - b1 ** model.t)) / (np.sqrt(Vv / (1 - b2 ** model.t)) + eps)
    model.b -= model.lr * g.sum()
    return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
# <<<END:loss>>>


# <<<BLOCK:train>>>
# 优化调度。候选进程看不到 official-valid/test 标签；checkpoint 采用固定训练预算。
# official validation 只在沙箱外由 trusted evaluator 用于模型选择和收敛判断。
HP = dict(k=16, lr=0.001, epochs=8, batch=8192)


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    m = FM(dim, k=HP['k'], lr=HP['lr'], seed=seed)
    rng = np.random.default_rng(seed)
    Xtr = Xs['train'][train_idx]
    for ep in range(1, HP['epochs'] + 1):
        idx = rng.permutation(len(y))
        for i in range(0, len(idx), HP['batch']):
            j = idx[i:i + HP['batch']]
            loss_and_step(m, Xtr[j], y[j])
        if verbose:
            print(f'  epoch {ep:2d}/{HP["epochs"]}', file=sys.stderr)
    return m, {'training_epochs': int(HP['epochs']),
               'checkpoint_policy': 'fixed_train_budget'}
# <<<END:train>>>


# <<<BLOCK:predict>>>
# 推理与输出。只写一个 float32 一维数组，长度必须等于该 split 的行数。
# row_id 由沙箱外的可信侧生成，候选接触不到提交格式。
def predict(model, Xs, split):
    return model.predict(Xs[split]).astype(np.float32)
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
