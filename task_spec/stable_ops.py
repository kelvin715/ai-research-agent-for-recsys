"""Read-only, tested primitives used by operator-mode candidate pipelines.

This module is mounted at ``/task`` with the data view.  The controller, rather
than the LLM, owns these implementations.  Candidate pipelines still pass the
normal static, smoke, resource, and output gates; the library merely removes
repeated handwritten implementations of standard recommendation operators.
"""
import numpy as np

import dataview as DV


BASE_FIELDS = ('user_id', 'video_id', 'author_id', 'tab', 'dur_bucket')
USER_CORE_COLUMNS = (0, 2, 3, 4, 5, 6, 7)
LGB_CATEGORICAL_FEATURES = (5,)
ENGAGEMENT_AUX_TARGETS = ('is_like', 'is_follow', 'is_comment', 'is_forward')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_data_view():
    splits = DV.load()
    return splits, np.arange(splits['train'].n)


def _user_gap_ms(rs):
    """Strictly earlier exposure gap within a split; no feedback is used."""
    order = np.lexsort((rs.time_ms, rs.user_id))
    users = np.asarray(rs.user_id)[order]
    times = np.asarray(rs.time_ms, dtype=np.float64)[order]
    previous = np.r_[0.0, times[:-1]]
    same_user = np.r_[False, users[1:] == users[:-1]]
    gap = np.where(same_user, np.maximum(0.0, times - previous), -1.0)
    out = np.empty(len(order), dtype=np.float64)
    out[order] = gap
    return out


def _raw_fields(rs, dur_edges, config, gap_edges):
    fields = [
        np.asarray(rs.user_id), np.asarray(rs.video_id), np.asarray(rs.author_id),
        np.asarray(rs.tab),
        np.searchsorted(dur_edges, rs.duration_ms).astype(np.int32),
    ]
    enabled = set(config.get('features', ()))
    if 'hour' in enabled:
        fields.append((np.asarray(rs.hourmin) // 100).astype(np.int32))
    if 'user_gap' in enabled:
        gap = _user_gap_ms(rs)
        fields.append(np.where(
            gap < 0, 0,
            1 + np.searchsorted(gap_edges, gap / 3_600_000.0),
        ).astype(np.int32))
    if 'user_core' in enabled:
        matrix = np.asarray(rs.user_categorical)
        fields.extend(matrix[:, index].astype(np.int32)
                      for index in USER_CORE_COLUMNS)
    return fields


def _fit_group_statistics(values, labels):
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    positive = np.bincount(
        inverse, weights=labels, minlength=len(unique)).astype(np.float64)
    return unique, count, positive


def _apply_group_statistics(values, fitted, global_mean, prior=20.0):
    unique, count, positive = fitted
    values = np.asarray(values)
    position = np.searchsorted(unique, values)
    clipped = np.clip(position, 0, max(len(unique) - 1, 0))
    hit = (position < len(unique)) & (unique[clipped] == values)
    n = np.where(hit, count[clipped], 0.0)
    pos = np.where(hit, positive[clipped], 0.0)
    rate = (pos + prior * global_mean) / (n + prior)
    return n, rate


def _group_features_from_values(values, train_idx, labels, global_mean, prior=20.0):
    """Train rows receive leave-one-out rates; evaluation rows receive full-train rates."""
    train_values = np.asarray(values['train'])
    fitted = _fit_group_statistics(train_values[train_idx], labels)
    output = {}
    for split, split_values in values.items():
        count, rate = _apply_group_statistics(
            split_values, fitted, global_mean, prior=prior)
        output[split] = [count, rate]

    unique, fitted_count, fitted_positive = fitted
    fit_values = train_values[train_idx]
    position = np.searchsorted(unique, fit_values)
    loo_count = np.maximum(fitted_count[position] - 1.0, 0.0)
    loo_positive = fitted_positive[position] - labels
    loo_rate = (loo_positive + prior * global_mean) / (loo_count + prior)
    output['train'][0][train_idx] = loo_count
    output['train'][1][train_idx] = loo_rate
    return output


def _causal_group_features_from_values(
        values, train_dates, train_idx, labels, global_mean, prior=20.0):
    """Strictly-prior-date train rates and full-selected-train evaluation rates.

    Row-wise leave-one-out target encoding is subtly anti-correlated with its own target:
    removing a positive lowers its rate while removing a negative raises it. Grouping selected
    training rows by (key, date) and using only previous dates avoids both that artifact and
    same-day/future leakage.
    """
    output = _group_features_from_values(
        values, train_idx, labels, global_mean, prior=prior)
    selected_values = np.asarray(values['train'])[train_idx]
    selected_dates = np.asarray(train_dates)[train_idx]
    labels = np.asarray(labels, dtype=np.float64)
    order = np.lexsort((selected_dates, selected_values))
    sorted_values = selected_values[order]
    sorted_dates = selected_dates[order]
    sorted_labels = labels[order]
    daily_start = np.r_[True, (sorted_values[1:] != sorted_values[:-1])
                        | (sorted_dates[1:] != sorted_dates[:-1])]
    starts = np.flatnonzero(daily_start)
    daily_count = np.diff(np.r_[starts, len(order)]).astype(np.float64)
    daily_positive = np.add.reduceat(sorted_labels, starts)
    daily_values = sorted_values[starts]
    value_start = np.r_[True, daily_values[1:] != daily_values[:-1]]
    value_starts = np.flatnonzero(value_start)
    value_group = np.cumsum(value_start) - 1

    cumulative_count = np.cumsum(daily_count)
    cumulative_positive = np.cumsum(daily_positive)
    count_base = cumulative_count[value_starts] - daily_count[value_starts]
    positive_base = (
        cumulative_positive[value_starts] - daily_positive[value_starts])
    previous_count = (
        cumulative_count - daily_count - count_base[value_group])
    previous_positive = (
        cumulative_positive - daily_positive - positive_base[value_group])
    daily_group = np.cumsum(daily_start) - 1
    sorted_previous_count = previous_count[daily_group]
    sorted_previous_rate = (
        previous_positive[daily_group] + prior * global_mean
    ) / (sorted_previous_count + prior)
    selected_count = np.empty(len(order), dtype=np.float64)
    selected_rate = np.empty(len(order), dtype=np.float64)
    selected_count[order] = sorted_previous_count
    selected_rate[order] = sorted_previous_rate
    output['train'][0][train_idx] = selected_count
    output['train'][1][train_idx] = selected_rate
    return output


def _group_features(splits, train_idx, key, labels, global_mean, prior=20.0):
    values = {split: np.asarray(getattr(rows, key))
              for split, rows in splits.items()}
    return _group_features_from_values(
        values, train_idx, labels, global_mean, prior=prior)


def _lightgbm_features(splits, train_idx):
    """Safe dense features for the standalone tree ranker.

    Item/author targets are fitted only on the selected train rows.  A selected train row
    never reads its own label because its aggregate is leave-one-out; valid/test use the
    complete selected-train aggregate.
    """
    labels = DV.assert_trainable(
        splits['train'].label[train_idx], where='LightGBM train aggregates')
    global_mean = float(labels.mean()) if len(labels) else 0.5
    keys = {
        'item': {name: np.asarray(rows.video_id) for name, rows in splits.items()},
        'author': {name: np.asarray(rows.author_id) for name, rows in splits.items()},
        'tag': {name: np.asarray(rows.video_categorical)[:, 5]
                for name, rows in splits.items()},
        'music': {name: np.asarray(rows.video_categorical)[:, 4]
                  for name, rows in splits.items()},
        'video_type': {name: np.asarray(rows.video_categorical)[:, 0]
                       for name, rows in splits.items()},
        'upload_type': {name: np.asarray(rows.video_categorical)[:, 1]
                        for name, rows in splits.items()},
    }

    target_names = ('is_click', 'is_like', 'is_follow', 'is_hate', 'play_time_ms')
    auxiliary = DV.train_targets(target_names)
    duration = np.asarray(splits['train'].duration_ms, dtype=np.float64)
    play = np.asarray(auxiliary['play_time_ms'], dtype=np.float64)
    play_ratio = np.clip(play / np.maximum(duration, 1.0), 0.0, 5.0)
    complete = (play >= np.maximum(duration, 1.0)).astype(np.float64)

    train_dates = np.asarray(splits['train'].date)

    def statistic(key, signal, prior):
        selected = np.asarray(signal, dtype=np.float64)[train_idx]
        mean = float(selected.mean()) if len(selected) else 0.0
        return _causal_group_features_from_values(
            keys[key], train_dates, train_idx, selected, mean, prior=prior)

    item_lv = _causal_group_features_from_values(
        keys['item'], train_dates, train_idx, labels, global_mean, prior=20.0)
    item_click = statistic('item', auxiliary['is_click'], 20.0)
    item_like = statistic('item', auxiliary['is_like'], 8.0)
    item_follow = statistic('item', auxiliary['is_follow'], 8.0)
    item_hate = statistic('item', auxiliary['is_hate'], 8.0)
    item_play = statistic('item', play_ratio, 20.0)
    item_complete = statistic('item', complete, 20.0)
    author_lv = _causal_group_features_from_values(
        keys['author'], train_dates, train_idx, labels, global_mean, prior=20.0)
    author_play = statistic('author', play_ratio, 20.0)
    tag_lv = _causal_group_features_from_values(
        keys['tag'], train_dates, train_idx, labels, global_mean, prior=20.0)
    music_lv = _causal_group_features_from_values(
        keys['music'], train_dates, train_idx, labels, global_mean, prior=20.0)
    video_type_lv = _causal_group_features_from_values(
        keys['video_type'], train_dates, train_idx, labels, global_mean, prior=20.0)
    upload_type_lv = _causal_group_features_from_values(
        keys['upload_type'], train_dates, train_idx, labels, global_mean, prior=20.0)
    output = {}
    for split, rows in splits.items():
        hour = np.asarray(rows.hourmin, dtype=np.float64) // 100
        duration_ms = np.asarray(rows.duration_ms, dtype=np.float64)
        video_num = np.asarray(rows.video_numeric, dtype=np.float64)
        width = np.maximum(video_num[:, 1], 0.0)
        height = np.maximum(video_num[:, 2], 1.0)
        output[split] = np.column_stack([
            duration_ms,
            np.log1p(np.maximum(duration_ms, 0.0)),
            hour,
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.asarray(rows.tab, dtype=np.float32),
            width / height, np.log1p(width), np.log1p(height),
            np.log1p(item_lv[split][0]), item_lv[split][1],
            item_click[split][1], item_like[split][1],
            item_follow[split][1], item_hate[split][1],
            item_play[split][1], item_complete[split][1],
            np.log1p(author_lv[split][0]), author_lv[split][1],
            author_play[split][1],
            np.log1p(tag_lv[split][0]), tag_lv[split][1],
            music_lv[split][1], video_type_lv[split][1],
            upload_type_lv[split][1],
        ]).astype(np.float32)
    return output, int(output['train'].shape[1])


def build_features(splits, train_idx, config):
    """Encode configured categorical fields using train-only vocabularies."""
    if config.get('model_family') == 'lightgbm_rank':
        return _lightgbm_features(splits, train_idx)
    tr = splits['train']
    n_dur = int(config.get('dur_buckets', 10))
    dur_edges = np.quantile(
        tr.duration_ms[train_idx], np.linspace(0, 1, n_dur + 1)[1:-1])
    train_gap = _user_gap_ms(tr)[train_idx] / 3_600_000.0
    seen_gap = train_gap[train_gap >= 0]
    gap_edges = (np.quantile(seen_gap, np.linspace(0, 1, 10)[1:-1])
                 if len(seen_gap) else np.array([], dtype=np.float64))

    train_raw = _raw_fields(tr, dur_edges, config, gap_edges)
    vocabs = [np.unique(column[train_idx]) for column in train_raw]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

    def encode(rs):
        raw = _raw_fields(rs, dur_edges, config, gap_edges)
        result = np.empty((rs.n, len(raw)), dtype=np.int32)
        for index, (column, vocab) in enumerate(zip(raw, vocabs)):
            position = np.searchsorted(vocab, column)
            clipped = np.clip(position, 0, max(len(vocab) - 1, 0))
            hit = (position < len(vocab)) & (vocab[clipped] == column)
            result[:, index] = np.where(hit, position, len(vocab)) + offsets[index]
        return result

    return ({name: encode(rows) for name, rows in splits.items()},
            int(sum(dimensions)))


def watch_time_targets(play_ms, duration_ms, scale=12.0):
    play = np.maximum(np.asarray(play_ms, dtype=np.float64), 0.0)
    duration = np.maximum(np.asarray(duration_ms, dtype=np.float64), 0.0)
    valid = duration > 0
    target = (np.log1p(play) / scale).astype(np.float32)
    lower = (np.log1p(duration) / scale).astype(np.float32)
    censored = valid & (play >= duration)
    return target, lower, censored, valid


def build_target(splits, train_idx, config):
    main = splits['train'].label[train_idx].astype(np.float32)
    main = DV.assert_trainable(main, where='stable operator target')
    if config.get('objective') == 'pointwise_engagement_mtl':
        targets = DV.train_targets(ENGAGEMENT_AUX_TARGETS)
        auxiliary = np.column_stack([
            np.asarray(targets[name])[train_idx] for name in ENGAGEMENT_AUX_TARGETS
        ]).astype(np.float32)
        for index, name in enumerate(ENGAGEMENT_AUX_TARGETS):
            DV.assert_trainable(auxiliary[:, index], where=f'{name} auxiliary target')
        return {'main': main, 'engagement_aux': auxiliary}
    if config.get('objective') != 'bpr_censored_watch':
        return main
    play = DV.train_targets('play_time_ms')['play_time_ms'][train_idx]
    duration = splits['train'].duration_ms[train_idx]
    target, lower, censored, valid = watch_time_targets(play, duration)
    return {
        'main': main, 'watch_target': target, 'watch_lower': lower,
        'watch_censored': censored, 'watch_valid': valid,
    }


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, auxiliary=False):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = float(lr), float(l2)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        self.auxiliary = bool(auxiliary)
        if self.auxiliary:
            self.Wa = np.zeros(dim, dtype=np.float32)
            self.ba = np.float32(0.0)
            self.mWa = np.zeros_like(self.Wa); self.vWa = np.zeros_like(self.Wa)

    def logits(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(1)
        interaction = 0.5 * (
            (summed ** 2).sum(1) - (embeddings ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + interaction, embeddings, summed

    def auxiliary_logits(self, X, embeddings=None, summed=None):
        if embeddings is None or summed is None:
            _, embeddings, summed = self.logits(X)
        interaction = 0.5 * (
            (summed ** 2).sum(1) - (embeddings ** 2).sum((1, 2)))
        return self.ba + self.Wa[X].sum(1) + interaction

    def predict(self, X, batch=200_000):
        return np.concatenate([
            self.logits(X[start:start + batch])[0]
            for start in range(0, len(X), batch)
        ])

    def _adam_main(self, grad_v, grad_w, grad_b):
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        correction1 = 1 - beta1 ** self.t
        correction2 = 1 - beta2 ** self.t
        for parameter, gradient, mean, variance in (
                (self.V, grad_v, self.mV, self.vV),
                (self.W, grad_w, self.mW, self.vW)):
            mean *= beta1; mean += (1 - beta1) * gradient
            variance *= beta2; variance += (1 - beta2) * gradient * gradient
            parameter -= self.lr * (mean / correction1) / (
                np.sqrt(variance / correction2) + epsilon)
        self.b -= self.lr * grad_b

    def _adam_auxiliary(self, grad_w, grad_b):
        grad_w += self.l2 * self.Wa
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        correction1 = 1 - beta1 ** self.t
        correction2 = 1 - beta2 ** self.t
        self.mWa *= beta1; self.mWa += (1 - beta1) * grad_w
        self.vWa *= beta2; self.vWa += (1 - beta2) * grad_w * grad_w
        self.Wa -= self.lr * (self.mWa / correction1) / (
            np.sqrt(self.vWa / correction2) + epsilon)
        self.ba -= self.lr * grad_b

    def step_pointwise(self, X, y):
        count = len(y)
        logits, embeddings, summed = self.logits(X)
        probability = sigmoid(logits)
        gradient = ((probability - y) / count).astype(np.float32)
        grad_v = np.zeros_like(self.V); grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, X, gradient[:, None])
        np.add.at(grad_v, X, gradient[:, None, None]
                  * (summed[:, None, :] - embeddings))
        self._adam_main(grad_v, grad_w, gradient.sum())
        return float(-np.mean(y * np.log(probability + 1e-9)
                              + (1 - y) * np.log(1 - probability + 1e-9)))

    def step_bpr(self, positive, negative):
        count = len(positive)
        pos_logit, pos_e, pos_s = self.logits(positive)
        neg_logit, neg_e, neg_s = self.logits(negative)
        difference = pos_logit - neg_logit
        pos_gradient = ((sigmoid(difference) - 1.0) / count).astype(np.float32)
        neg_gradient = -pos_gradient
        grad_v = np.zeros_like(self.V); grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, positive, pos_gradient[:, None])
        np.add.at(grad_w, negative, neg_gradient[:, None])
        np.add.at(grad_v, positive, pos_gradient[:, None, None]
                  * (pos_s[:, None, :] - pos_e))
        np.add.at(grad_v, negative, neg_gradient[:, None, None]
                  * (neg_s[:, None, :] - neg_e))
        self._adam_main(grad_v, grad_w, 0.0)
        return float(np.logaddexp(0.0, -difference).mean())

    def step_bpr_censored(self, positive, negative, watch, aux_weight):
        count = len(positive)
        pos_logit, pos_e, pos_s = self.logits(positive)
        neg_logit, neg_e, neg_s = self.logits(negative)
        difference = pos_logit - neg_logit
        pos_gradient = ((sigmoid(difference) - 1.0) / count).astype(np.float32)
        neg_gradient = -pos_gradient

        pos_aux = self.auxiliary_logits(positive, pos_e, pos_s)
        neg_aux = self.auxiliary_logits(negative, neg_e, neg_s)
        target, lower, censored, valid, pos_index, neg_index = watch

        def auxiliary_gradient(prediction, indices):
            ok = valid[indices]
            exact = ok & (~censored[indices])
            right = ok & censored[indices]
            gradient = np.zeros(len(indices), dtype=np.float32)
            gradient[exact] = prediction[exact] - target[indices][exact]
            active = right & (prediction < lower[indices])
            gradient[active] = prediction[active] - lower[indices][active]
            return (np.float32(aux_weight) * gradient / count).astype(np.float32)

        pos_aux_gradient = auxiliary_gradient(pos_aux, pos_index)
        neg_aux_gradient = auxiliary_gradient(neg_aux, neg_index)
        grad_v = np.zeros_like(self.V); grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, positive, pos_gradient[:, None])
        np.add.at(grad_w, negative, neg_gradient[:, None])
        np.add.at(grad_v, positive, (pos_gradient + pos_aux_gradient)[:, None, None]
                  * (pos_s[:, None, :] - pos_e))
        np.add.at(grad_v, negative, (neg_gradient + neg_aux_gradient)[:, None, None]
                  * (neg_s[:, None, :] - neg_e))
        self._adam_main(grad_v, grad_w, 0.0)

        grad_aux = np.zeros_like(self.Wa)
        np.add.at(grad_aux, positive, pos_aux_gradient[:, None])
        np.add.at(grad_aux, negative, neg_aux_gradient[:, None])
        self._adam_auxiliary(
            grad_aux, float(pos_aux_gradient.sum() + neg_aux_gradient.sum()))
        return float(np.logaddexp(0.0, -difference).mean())


class LightGBMRanker:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, features, batch=200_000):
        return self.booster.predict(features)


class TorchDeepFMMTL:
    """CPU-only DeepFM with four train-only engagement auxiliary heads."""

    def __init__(self, dimension, field_count, hp, seed):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                'deepfm_engagement_mtl_v1 requires the pinned CPU PyTorch environment; '
                'rebuild it with trusted/make_venv.sh') from exc

        threads = max(1, int(hp.get('torch_threads', 1)))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows this setting only before inter-op work starts.  The sandbox is
            # already process-isolated and OMP/MKL are pinned to one thread as a backstop.
            pass
        torch.manual_seed(int(seed))
        self.torch = torch
        self.device = torch.device('cpu')
        self.aux_weight = float(hp.get('aux_weight', 0.2))
        k = int(hp.get('k', 16))
        hidden = [int(value) for value in hp.get('hidden', (128, 64))]
        if not hidden or any(value <= 0 for value in hidden):
            raise ValueError('DeepFM hidden dimensions must be positive')

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(int(dimension), k)
                nn.init.normal_(self.embedding.weight, 0.0, 0.01)
                self.linear = nn.Embedding(int(dimension), 1)
                nn.init.zeros_(self.linear.weight)
                self.bias = nn.Parameter(torch.zeros(1))
                sizes = [int(field_count) * k] + hidden
                layers = []
                for input_size, output_size in zip(sizes[:-1], sizes[1:]):
                    layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
                self.deep = nn.Sequential(*layers)
                self.main_head = nn.Linear(sizes[-1], 1)
                self.auxiliary_heads = nn.Linear(sizes[-1], len(ENGAGEMENT_AUX_TARGETS))

            def forward(self, x):
                embeddings = self.embedding(x)
                summed = embeddings.sum(1)
                interaction = 0.5 * (
                    (summed ** 2).sum(1) - (embeddings ** 2).sum((1, 2)))
                linear = self.linear(x).sum((1, 2))
                batch_size, fields, factors = embeddings.shape
                deep = self.deep(embeddings.reshape(batch_size, fields * factors))
                main = self.bias + linear + interaction + self.main_head(deep).squeeze(-1)
                return main, self.auxiliary_heads(deep)

        self.network = Network().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=float(hp.get('lr', 0.001)),
            weight_decay=float(hp.get('l2', 1e-4)))
        self.loss = nn.BCEWithLogitsLoss()

    def step(self, features, main_target, auxiliary_target):
        torch = self.torch
        self.network.train()
        x = torch.as_tensor(features, dtype=torch.long, device=self.device)
        main = torch.as_tensor(main_target, dtype=torch.float32, device=self.device)
        auxiliary = torch.as_tensor(
            auxiliary_target, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        main_logits, auxiliary_logits = self.network(x)
        main_loss = self.loss(main_logits, main)
        auxiliary_loss = self.loss(auxiliary_logits, auxiliary)
        total = main_loss + self.aux_weight * auxiliary_loss
        total.backward()
        self.optimizer.step()
        return float(main_loss.detach().cpu())

    def predict(self, features, batch=200_000):
        torch = self.torch
        self.network.eval()
        output = []
        with torch.inference_mode():
            for start in range(0, len(features), batch):
                x = torch.as_tensor(
                    features[start:start + batch], dtype=torch.long, device=self.device)
                logits, _ = self.network(x)
                output.append(logits.cpu().numpy())
        if not output:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(output).astype(np.float32, copy=False)


def loss_and_step(model, *args):
    """Compatibility hook for the pipeline's named loss block."""
    if len(args) == 2:
        return model.step_pointwise(*args)
    raise ValueError('stable operator loss is scheduled by stable_ops.train')


def _pair_pools(users, labels):
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    positives, negatives = [], []
    binary = labels > 0.5
    for start, end in zip(starts, ends):
        block = order[start:end]
        mask = binary[block]
        if mask.any() and (~mask).any():
            positives.append(block[mask])
            negatives.append(block[~mask])
    return positives, negatives


def _group_sizes(sorted_users, sorted_dates=None):
    sorted_users = np.asarray(sorted_users)
    change = sorted_users[1:] != sorted_users[:-1]
    if sorted_dates is not None:
        sorted_dates = np.asarray(sorted_dates)
        change |= sorted_dates[1:] != sorted_dates[:-1]
    boundaries = np.r_[0, np.flatnonzero(change) + 1, len(sorted_users)]
    return np.diff(boundaries).astype(np.int32)


def _train_lightgbm_ranker(splits, train_idx, features, labels, seed, config):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            'item_lambdarank requires the pre-run candidate environment to include lightgbm; '
            'rebuild it with trusted/make_venv.sh') from exc

    train_users = np.asarray(splits['train'].user_id)[train_idx]
    train_dates = np.asarray(splits['train'].date)[train_idx]
    order = np.lexsort((train_dates, train_users))
    train_x = np.asarray(features['train'])[train_idx][order]
    train_y = np.asarray(labels, dtype=np.float32)[order]
    sorted_users = train_users[order]
    sorted_dates = train_dates[order]
    dataset = lgb.Dataset(
        train_x, label=train_y, group=_group_sizes(sorted_users, sorted_dates),
        categorical_feature=list(LGB_CATEGORICAL_FEATURES), free_raw_data=False)
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [5],
        'lambdarank_truncation_level': 10,
        'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 50,
        'feature_fraction': 0.85, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'lambda_l2': 1.0, 'verbosity': -1, 'num_threads': 1,
        'force_row_wise': True,
        'seed': int(seed), 'bagging_seed': int(seed),
        'feature_fraction_seed': int(seed),
    }
    rounds = int(config['hp'].get('num_boost_round', 300))
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    return LightGBMRanker(booster), {
        'training_epochs': rounds,
        'checkpoint_policy': 'fixed_train_only_boosting_rounds',
        'model_family': 'lightgbm_rank',
        'feature_count': int(train_x.shape[1]),
    }


def train(splits, train_idx, features, dimension, target, seed, config,
          verbose=False):
    hp = config['hp']
    objective = config.get('objective', 'pointwise')
    main = target['main'] if isinstance(target, dict) else target
    if config.get('model_family') == 'lightgbm_rank':
        return _train_lightgbm_ranker(
            splits, train_idx, features, main, seed, config)
    if config.get('model_family') == 'torch_deepfm_mtl':
        model = TorchDeepFMMTL(
            dimension, int(np.asarray(features['train']).shape[1]), hp, seed)
        rng = np.random.default_rng(seed)
        train_x = np.asarray(features['train'])[train_idx]
        auxiliary = np.asarray(target['engagement_aux'], dtype=np.float32)
        epochs, batch = int(hp['epochs']), int(hp['batch'])
        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(len(main))
            for start in range(0, len(permutation), batch):
                indices = permutation[start:start + batch]
                model.step(train_x[indices], main[indices], auxiliary[indices])
            if verbose:
                print(f'  epoch {epoch:2d}/{epochs} device=cpu',
                      file=__import__('sys').stderr)
        return model, {
            'training_epochs': epochs,
            'checkpoint_policy': 'fixed_train_only_public_prior_epoch_12',
            'model_family': 'torch_deepfm_mtl',
            'device': 'cpu',
            'feature_count': int(np.asarray(features['train']).shape[1]),
            'embedding_dimension': int(dimension),
            'auxiliary_targets': list(ENGAGEMENT_AUX_TARGETS),
            'operator_hp': dict(hp),
        }

    auxiliary = objective == 'bpr_censored_watch'
    model = FM(dimension, k=int(hp['k']), lr=float(hp['lr']),
               l2=float(hp.get('l2', 1e-6)), seed=seed, auxiliary=auxiliary)
    rng = np.random.default_rng(seed)
    train_x = features['train'][train_idx]
    epochs, batch = int(hp['epochs']), int(hp['batch'])

    if objective == 'pointwise':
        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(len(main))
            for start in range(0, len(permutation), batch):
                indices = permutation[start:start + batch]
                model.step_pointwise(train_x[indices], main[indices])
            if verbose:
                print(f'  epoch {epoch:2d}/{epochs}', file=__import__('sys').stderr)
    else:
        users = np.asarray(splits['train'].user_id)[train_idx]
        positive_blocks, negative_pools = _pair_pools(users, main)
        n_neg = int(hp.get('n_neg', 1))
        positive_all = np.concatenate([
            np.tile(block, n_neg) for block in positive_blocks
        ]) if positive_blocks else np.array([], dtype=np.int32)
        counts = [len(block) * n_neg for block in positive_blocks]
        for epoch in range(1, epochs + 1):
            if not len(positive_all):
                break
            negative_all = np.concatenate([
                rng.choice(pool, size=count, replace=True)
                for pool, count in zip(negative_pools, counts)
            ])
            permutation = rng.permutation(len(positive_all))
            for start in range(0, len(permutation), batch):
                selected = permutation[start:start + batch]
                pos_index = positive_all[selected]
                neg_index = negative_all[selected]
                if auxiliary:
                    watch = (
                        target['watch_target'], target['watch_lower'],
                        target['watch_censored'], target['watch_valid'],
                        pos_index, neg_index,
                    )
                    model.step_bpr_censored(
                        train_x[pos_index], train_x[neg_index], watch,
                        float(hp.get('aux_weight', 0.2)))
                else:
                    model.step_bpr(train_x[pos_index], train_x[neg_index])
            if verbose:
                print(f'  epoch {epoch:2d}/{epochs} pairs={len(positive_all)}',
                      file=__import__('sys').stderr)

    return model, {
        'training_epochs': epochs,
        'checkpoint_policy': 'fixed_train_budget',
        'operator_objective': objective,
        'operator_features': list(config.get('features', ())),
        'operator_hp': dict(hp),
    }


def predict(model, features, split):
    return model.predict(features[split]).astype(np.float32)
