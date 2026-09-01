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
    base = SO.build_target(splits, train_idx, OP_CONFIG)
    train = splits['train']

    ratio, valid = DV.watch_ratio()
    ratio = np.asarray(ratio, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    duration_ms = np.asarray(train.duration_ms, dtype=np.float32)

    safe_ratio = np.clip(ratio, 0.0, 1.0)
    safe_duration = np.maximum(duration_ms, 0.0)
    watch_ms = safe_ratio * safe_duration

    watch_t = (np.log1p(watch_ms) / 12.0).astype(np.float32)
    dur_t = (np.log1p(safe_duration) / 12.0).astype(np.float32)

    complete = valid & (safe_ratio >= 0.95)
    incomplete = valid & (~complete)

    out = dict(base)
    out['watch_time_target'] = watch_t
    out['watch_time_valid'] = valid.astype(np.float32)
    out['watch_time_complete'] = complete.astype(np.float32)
    out['watch_time_incomplete'] = incomplete.astype(np.float32)
    out['watch_time_duration_target'] = dur_t
    return out
# <<<END:target>>>


# <<<BLOCK:model>>>
sigmoid = SO.sigmoid
FM = SO.FM

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFMMTLWatch(nn.Module):
    def __init__(self, dim, field_count, k=16, hidden=(128, 64)):
        super().__init__()
        self.dim = int(dim)
        self.field_count = int(field_count)
        self.k = int(k)
        self.hidden = list(hidden)

        self.bias = nn.Parameter(torch.zeros(1))
        self.lin = nn.Embedding(self.dim, 1)
        self.emb = nn.Embedding(self.dim, self.k)

        layers = []
        in_dim = self.field_count * self.k
        for h in self.hidden:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        self.mlp = nn.Sequential(*layers)

        self.main_head = nn.Linear(in_dim, 1)
        self.aux_head = nn.Linear(in_dim, 1)
        self.watch_head = nn.Linear(in_dim, 1)

        nn.init.zeros_(self.lin.weight)
        nn.init.normal_(self.emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.main_head.weight)
        nn.init.zeros_(self.main_head.bias)
        nn.init.xavier_uniform_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)
        nn.init.xavier_uniform_(self.watch_head.weight)
        nn.init.zeros_(self.watch_head.bias)

    def forward(self, x):
        x = x.long()
        lin_term = self.lin(x).sum(dim=1).squeeze(1)
        v = self.emb(x)
        summed = v.sum(dim=1)
        fm_term = 0.5 * ((summed * summed) - (v * v).sum(dim=1)).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1))
        main = self.bias + lin_term + fm_term + self.main_head(deep).squeeze(1)
        aux = self.aux_head(deep).squeeze(1)
        watch = self.watch_head(deep).squeeze(1)
        return {'main': main, 'aux': aux, 'watch': watch}

    def predict_main(self, x):
        return self.forward(x)['main']
# <<<END:model>>>


# <<<BLOCK:loss>>>
loss_and_step = SO.loss_and_step
# <<<END:loss>>>


# <<<BLOCK:train>>>
HP = dict(OP_CONFIG['hp'])


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    import copy
    import torch
    import torch.nn.functional as F

    hp = dict(HP)
    torch.set_num_threads(int(hp['torch_threads']))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    x_train = np.asarray(Xs['train'][train_idx], dtype=np.int64)
    y_main = np.asarray(y['main'][train_idx], dtype=np.float32)
    y_aux = np.asarray(y['engagement_aux'][train_idx], dtype=np.float32)
    wt = np.asarray(y['watch_time_target'][train_idx], dtype=np.float32)
    wt_complete = np.asarray(y['watch_time_complete'][train_idx], dtype=np.float32)
    wt_incomplete = np.asarray(y['watch_time_incomplete'][train_idx], dtype=np.float32)
    wt_dur = np.asarray(y['watch_time_duration_target'][train_idx], dtype=np.float32)

    DV.assert_trainable(y_main, 'train.main')
    DV.assert_trainable(y_aux, 'train.engagement_aux')

    field_count = int(x_train.shape[1])
    model = DeepFMMTLWatch(dim=dim, field_count=field_count, k=hp['k'], hidden=hp['hidden'])
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=float(hp['l2']))

    batch_size = int(hp['batch'])
    aux_weight = float(hp['aux_weight'])
    watch_weight = 0.05
    n = len(train_idx)

    x_tensor = torch.from_numpy(x_train)
    y_main_t = torch.from_numpy(y_main)
    y_aux_t = torch.from_numpy(y_aux)
    wt_t = torch.from_numpy(wt)
    wt_complete_t = torch.from_numpy(wt_complete)
    wt_incomplete_t = torch.from_numpy(wt_incomplete)
    wt_dur_t = torch.from_numpy(wt_dur)

    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(int(hp['epochs'])):
        perm = rng.permutation(n)
        sum_main = 0.0
        sum_aux = 0.0
        sum_watch = 0.0
        sum_total = 0.0
        seen = 0

        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_tensor[idx]
            main_b = y_main_t[idx]
            aux_b = y_aux_t[idx]
            wt_b = wt_t[idx]
            complete_b = wt_complete_t[idx]
            incomplete_b = wt_incomplete_t[idx]
            dur_b = wt_dur_t[idx]

            out = model(xb)
            main_loss = F.binary_cross_entropy_with_logits(out['main'], main_b)
            aux_loss = F.binary_cross_entropy_with_logits(out['aux'], aux_b)

            watch_pred = out['watch']
            inc_mask = incomplete_b > 0.5
            comp_mask = complete_b > 0.5

            inc_count = int(inc_mask.sum().item())
            comp_count = int(comp_mask.sum().item())
            valid_count = inc_count + comp_count

            if inc_count > 0:
                inc_loss = F.smooth_l1_loss(watch_pred[inc_mask], wt_b[inc_mask], reduction='mean')
            else:
                inc_loss = watch_pred.sum() * 0.0

            if comp_count > 0:
                hinge = F.relu(dur_b[comp_mask] - watch_pred[comp_mask])
                comp_loss = (hinge * hinge).mean()
            else:
                comp_loss = watch_pred.sum() * 0.0

            if valid_count > 0:
                watch_loss = (inc_loss * inc_count + comp_loss * comp_count) / float(valid_count)
            else:
                watch_loss = watch_pred.sum() * 0.0

            loss = main_loss + aux_weight * aux_loss + watch_weight * watch_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = len(idx)
            seen += bs
            sum_main += float(main_loss.detach()) * bs
            sum_aux += float(aux_loss.detach()) * bs
            sum_watch += float(watch_loss.detach()) * bs
            sum_total += float(loss.detach()) * bs

        rec = {
            'epoch': epoch + 1,
            'main_loss': sum_main / max(seen, 1),
            'aux_loss': sum_aux / max(seen, 1),
            'watch_loss': sum_watch / max(seen, 1),
            'total_loss': sum_total / max(seen, 1),
        }
        history.append(rec)
        if rec['total_loss'] < history[best_epoch]['total_loss'] if best_epoch >= 0 else True:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(json.dumps(rec))

    model.load_state_dict(best_state)

    info = {
        'model_family': 'torch_deepfm_mtl_watch',
        'objective': 'pointwise_engagement_mtl_plus_censored_watch',
        'epochs': int(hp['epochs']),
        'batch': batch_size,
        'lr': float(hp['lr']),
        'l2': float(hp['l2']),
        'k': int(hp['k']),
        'hidden': list(hp['hidden']),
        'aux_weight': aux_weight,
        'watch_weight': watch_weight,
        'checkpoint_policy': 'train_only_best_total_loss_epoch',
        'best_epoch': int(best_epoch + 1),
        'history': history,
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
