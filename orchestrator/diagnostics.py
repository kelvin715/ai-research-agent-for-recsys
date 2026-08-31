"""Deterministic train-only observations supplied to the research agent."""
import os

import numpy as np


FEATURES = ('user_id', 'video_id', 'author_id', 'tab', 'duration_ms',
            'date', 'hourmin', 'time_ms')
TARGETS = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
           'is_hate', 'long_view', 'is_profile_enter')


def _load(parsed_dir, name):
    return np.load(os.path.join(parsed_dir, name + '.npy'), mmap_mode='r')


def build(parsed_dir):
    """Summarize only train labels/targets; valid/test contribute feature coverage only."""
    train_y = np.asarray(_load(parsed_dir, 'train.label'), dtype=np.float64)
    result = {
        'policy': 'labels and post-exposure targets are train-only',
        'splits': {},
        'train_feature_cardinality': {},
        'train_target_rates': {},
        'train_target_phi_with_long_view': {},
    }
    for split in ('train', 'valid', 'test'):
        u = _load(parsed_dir, f'{split}.user_id')
        v = _load(parsed_dir, f'{split}.video_id')
        result['splits'][split] = {
            'rows': int(len(u)), 'users': int(len(np.unique(u))),
            'videos': int(len(np.unique(v))),
        }
    for name in FEATURES:
        arr = _load(parsed_dir, f'train.{name}')
        result['train_feature_cardinality'][name] = {
            'distinct': int(len(np.unique(arr))),
            'min': float(np.min(arr)), 'max': float(np.max(arr)),
        }
    for name in TARGETS:
        target = np.asarray(_load(parsed_dir, f'train.target.{name}'), dtype=np.float64)
        result['train_target_rates'][name] = float(target.mean())
        denom = target.std() * train_y.std()
        result['train_target_phi_with_long_view'][name] = (
            float(np.mean((target - target.mean()) * (train_y - train_y.mean())) / denom)
            if denom > 0 else None)
    duration = _load(parsed_dir, 'train.duration_ms')
    result['train_duration_zero_rows'] = int(np.count_nonzero(duration == 0))
    return result
