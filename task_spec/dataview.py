"""候选 pipeline 的唯一数据入口。只读，由 orchestrator 挂载到 /task，不可修改。

为什么必须走这里而不是自己读 CSV：
  - 反馈信号（play_time_ms / is_click / ...）只能当**训练目标**，不能当任何 split 的输入特征。
    long_view ≈ f(play_time_ms, duration_ms)，把它当特征等于把答案喂给模型。
  - train_targets() 在构造上只返回 train 行；评测行的反馈值在磁盘上根本不存在。
    所以「把 play_time_ms 当特征」这条路的结局是推理时 KeyError，是自然 tripwire。

标签边界：候选视图只包含 train.label。official-valid/test 标签只存在于沙箱外的
trusted evaluator；访问 RowSet('valid').label 或 RowSet('test').label 会响亮地失败。
"""
import os

import numpy as np

DATA_ROOT = os.environ.get('TRACK2_DATA', '/data')
WITHHELD = -1

SCALAR_FEATURE_COLS = ('user_id', 'video_id', 'author_id', 'tab',
                       'duration_ms', 'date', 'hourmin', 'time_ms')

USER_CATEGORICAL_NAMES = (
    'user_active_degree', 'is_lowactive_period', 'is_live_streamer',
    'is_video_author', 'follow_user_num_range', 'fans_user_num_range',
    'friend_user_num_range', 'register_days_range',
) + tuple(f'onehot_feat{i}' for i in range(18))
USER_NUMERIC_NAMES = (
    'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
)
VIDEO_CATEGORICAL_NAMES = (
    'video_type', 'upload_type', 'visible_status', 'music_id', 'music_type', 'tag',
)
VIDEO_NUMERIC_NAMES = (
    'video_duration', 'server_width', 'server_height', 'upload_day',
)
MATRIX_FEATURE_COLS = (
    'user_categorical', 'user_numeric', 'video_categorical', 'video_numeric',
)
FEATURE_COLS = SCALAR_FEATURE_COLS + MATRIX_FEATURE_COLS

TARGET_COLS = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
               'is_hate', 'long_view', 'play_time_ms', 'profile_stay_time',
               'comment_stay_time', 'is_profile_enter')

SPLITS = ('train', 'valid', 'test')

_cache = {}


def _parsed(name):
    if name not in _cache:
        path = os.path.join(DATA_ROOT, 'parsed', f'{name}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{name}.npy 不存在。若你在取评测集的反馈信号，那是设计使然：'
                f'它们只对 train 存在。')
        _cache[name] = np.load(path, mmap_mode='r')
    return _cache[name]


class RowSet:
    """一个 split 的所有行，按 manifest 规范行序。列是只读 numpy 数组。"""

    __slots__ = ('split', '_cols', '_label', 'n')

    def __init__(self, split):
        if split not in SPLITS:
            raise ValueError(f'未知 split: {split}')
        self.split = split
        self._cols = {c: np.asarray(_parsed(f'{split}.{c}')) for c in FEATURE_COLS}
        self._label = (np.asarray(_parsed('train.label'))
                       if split == 'train' else None)
        self.n = len(self._cols['user_id'])

    @property
    def label(self):
        if self._label is None:
            raise AttributeError(
                f'{self.split}.label 已从候选数据视图物理扣留；'
                'official validation 只能由沙箱外 trusted evaluator 使用。')
        return self._label

    def __getattr__(self, name):
        if name == 'label' and self._label is None:
            raise AttributeError(
                f'{self.split}.label 已从候选数据视图物理扣留；'
                'official validation 只能由沙箱外 trusted evaluator 使用。')
        if name in self._cols:
            return self._cols[name]
        raise AttributeError(
            f'{name!r} 不是可用于推理的特征。可用: {list(FEATURE_COLS)}。'
            f'若 {name!r} 是曝光后反馈信号，只能经 train_targets() 当训练目标。')

    def __len__(self):
        return self.n

    @property
    def has_labels(self):
        return self._label is not None

    def __repr__(self):
        return (f'<RowSet {self.split} n={self.n:,d} '
                f'labels={"有" if self.has_labels else "已扣留"}>')


def load(split=None):
    """load() 返回三个 split 的 dict；load('train') 返回单个 RowSet。"""
    if split is None:
        return {s: RowSet(s) for s in SPLITS}
    return RowSet(split)


def train_targets(names):
    """取 train 行的曝光后反馈信号，用作训练/辅助目标。

    只有 train。传入其它 split 没有对应参数 —— 这不是检查，是这些文件不存在。
    """
    if isinstance(names, str):
        names = [names]
    out = {}
    for name in names:
        if name not in TARGET_COLS:
            raise ValueError(f'{name!r} 不是反馈信号。可用: {list(TARGET_COLS)}')
        out[name] = np.asarray(_parsed(f'train.target.{name}'))
    return out


def watch_ratio():
    """train-only 的观看完成比：min(play_time, req) / req，req = min(duration, 18000)。

    返回 (ratio, valid)。调用方必须使用 valid 掩码，因为 duration_ms == 0 时
    完成比无定义。该函数只提供数据变换，不携带任何人工实验结论。
    """
    tr = RowSet('train')
    play = train_targets('play_time_ms')['play_time_ms'].astype(np.float64)
    dur = tr.duration_ms.astype(np.float64)
    valid = dur > 0
    req = np.minimum(np.where(valid, dur, 1.0), 18000.0)
    ratio = np.clip(np.minimum(play, req) / req, 0.0, 1.0).astype(np.float32)
    ratio[~valid] = np.nan
    return ratio, valid


def user_history():
    """按用户分组的 train 期交互序列，供序列建模（DIN/SIM 类）使用。

    只取 train 行，所以对 valid/test 而言整段历史都在过去，不存在前视泄漏。
    对 train 行自身建特征时，调用方需自己用返回的 date/time_ms 施加 cutoff。

    返回 (order, starts) —— order 是按 (user_id, time_ms) 排序后的 train 行下标，
    starts 是每个 user 在 order 里的起止，用 np.searchsorted 查。
    """
    tr = RowSet('train')
    order = np.lexsort((tr.time_ms, tr.user_id))
    uid_sorted = tr.user_id[order]
    uniq, starts = np.unique(uid_sorted, return_index=True)
    return order, uniq, starts


def assert_trainable(y, where='loss'):
    """守卫：标签被扣留的行不得进入训练或评估。

    没有这个守卫，误用 test 标签会静默产生一个「看起来合理」的分数 —— 那是最坏的失败模式。
    """
    y = np.asarray(y)
    n_bad = int((y == WITHHELD).sum())
    if n_bad:
        raise ValueError(
            f'{where}: 有 {n_bad:,d} 行的标签被扣留（哨兵 {WITHHELD}）。'
            f'test 标签由组委会持有，候选进程不应也无法访问。')
    if not np.isfinite(y).all():
        raise ValueError(f'{where}: 标签含 NaN/Inf')
    if not np.isin(y, (0, 1)).all():
        raise ValueError(f'{where}: 二元标签必须只含 0/1')
    return y


def eval_inputs(user_id, label):
    """把 validation 列转成官方 evaluate() 可安全接收的 Python 标量序列。"""
    label = assert_trainable(label, where='validation metric')
    return (np.asarray(user_id).tolist(),
            np.asarray(label).astype(np.int64).tolist())


def n_rows(split):
    return len(_parsed(f'{split}.user_id'))


def describe():
    """给 task_spec 用的数据摘要，也是喂给 LLM 的那份（不含任何分数）。"""
    lines = []
    for s in SPLITS:
        rs = RowSet(s)
        lines.append(f'{s:5s} n={rs.n:>9,d}  users={len(np.unique(rs.user_id)):>6,d}  '
                     f'videos={len(np.unique(rs.video_id)):>5,d}  '
                     f'labels={"有" if rs.has_labels else "已扣留"}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe())
