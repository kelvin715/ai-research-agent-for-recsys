"""列角色的单一事实来源。

其它模块（make_views / gates / dataview）一律从这里取，不得各自硬编码列名 ——
否则白名单和脱敏逻辑会漂移，而漂移的方向永远是「泄漏」。

角色定义（对应 TRACK2-AGENT-EXPERIMENT-PLAN.md §4.1 G2）：
  INFERENCE_FEATURE : 三个 split 都可见，可以进特征
  TRAIN_ONLY_TARGET : 曝光后才产生的反馈信号；只有 train 行有值，只能当训练目标
  LABEL             : 官方标签，本身也是 TRAIN_ONLY_TARGET（validation 保留用于开发/早停）
  META              : 结构性列，不进模型
"""

LABEL = 'long_view'

# 日志 CSV 的 19 列（log_standard_*.csv）
LOG_INFERENCE_FEATURE = [
    'user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'duration_ms', 'tab',
]

# Static side information shipped with KuaiRand-Pure. These fields exist before
# an impression, so they are legitimate inference inputs. make_views encodes
# categorical strings deterministically without consulting interaction labels.
USER_CATEGORICAL_FEATURE = [
    'user_active_degree', 'is_lowactive_period', 'is_live_streamer',
    'is_video_author', 'follow_user_num_range', 'fans_user_num_range',
    'friend_user_num_range', 'register_days_range',
] + [f'onehot_feat{i}' for i in range(18)]
USER_NUMERIC_FEATURE = [
    'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
]
VIDEO_CATEGORICAL_FEATURE = [
    'video_type', 'upload_type', 'visible_status', 'music_id', 'music_type', 'tag',
]
VIDEO_NUMERIC_FEATURE = [
    'video_duration', 'server_width', 'server_height', 'upload_day',
]

INFERENCE_FEATURE = (
    LOG_INFERENCE_FEATURE + USER_CATEGORICAL_FEATURE + USER_NUMERIC_FEATURE
    + VIDEO_CATEGORICAL_FEATURE + VIDEO_NUMERIC_FEATURE
)

# 曝光后信号。这些列在 test 行会被 make_views 置空；validation 按项目规则可用于开发。
TRAIN_ONLY_TARGET = [
    'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
    'long_view', 'play_time_ms', 'profile_stay_time', 'comment_stay_time',
    'is_profile_enter',
]

META = ['is_rand']

# Candidate-facing matrices. Keeping their schemas here prevents preprocessing,
# the manifest, and dataview from silently drifting apart.
STATIC_ARRAYS = {
    'user_categorical': USER_CATEGORICAL_FEATURE,
    'user_numeric': USER_NUMERIC_FEATURE,
    'video_categorical': VIDEO_CATEGORICAL_FEATURE,
    'video_numeric': VIDEO_NUMERIC_FEATURE,
}

# 组委会未测过、统计窗口未知、有时间泄漏嫌疑（见 TRACK2-RESEARCH.md §2.6）。
# 默认不挂进沙箱；要用必须显式开启并在 README 声明。
QUARANTINED_FILES = ['video_features_statistic_pure.csv']

# 日期落在 valid+test 窗口，不得作训练数据（TRACK2-RESEARCH.md §2.6）
FORBIDDEN_TRAIN_FILES = ['log_random_4_22_to_5_08_pure.csv']

SPLITS = {
    'train': (20220408, 20220421),
    'valid': (20220422, 20220428),
    'test':  (20220429, 20220508),
}

# 标签被扣留的哨兵值。data.py 的 patch 用它，dataview 的 assert 用它。
WITHHELD = -1


def assert_disjoint():
    """列角色不能重叠 —— 重叠意味着某列既能当特征又能当目标。"""
    a, b, c = set(INFERENCE_FEATURE), set(TRAIN_ONLY_TARGET), set(META)
    assert not (a & b), f"特征与训练目标重叠: {a & b}"
    assert not (a & c), f"特征与 META 重叠: {a & c}"
    assert not (b & c), f"训练目标与 META 重叠: {b & c}"
    assert LABEL in b, "LABEL 必须属于 TRAIN_ONLY_TARGET"


assert_disjoint()
