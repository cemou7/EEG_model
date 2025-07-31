import os
import numpy as np
import random
from scipy.signal import butter, filtfilt

# === 配置 ===
mi_data_dir = 'data_2a_TE_no_reject'
rest_data_dir = 'dataset/data_2a_binary_MI_Rest'
save_dir = 'dataset/composite_mi_rest_data'
os.makedirs(save_dir, exist_ok=True)

selected_subjects = ['A01', 'A02', 'A03']  # 可修改被试编号
fs = 250
min_len_sec = 2
max_len_sec = 4

# === 滤波器：低通10Hz（用于拼接平滑） ===
def lowpass_filter(data, cutoff=10, fs=250, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=1)

# === 获取样本 ===
def load_data(subject, suffix):
    data_path = os.path.join(rest_data_dir, f"{subject}_{suffix}_data.npy")
    label_path = os.path.join(rest_data_dir, f"{subject}_{suffix}_label.npy")
    if not os.path.exists(data_path):
        return None, None
    X = np.load(data_path)
    y = np.load(label_path)
    return X, y

# === 主构造函数 ===
def create_samples(subject, num_samples=100):
    # 加载静息样本（label=0）与想象样本（来自 no_reject 文件夹）
    rest_X_T, rest_y_T = load_data(subject, 'T')
    rest_X_E, rest_y_E = load_data(subject, 'E')
    rest_X = np.concatenate([rest_X_T[rest_y_T == 0], rest_X_E[rest_y_E == 0]], axis=0)

    mi_X_T = np.load(os.path.join(mi_data_dir, f"{subject}_T_data.npy"))
    mi_y_T = np.load(os.path.join(mi_data_dir, f"{subject}_T_label.npy"))
    mi_X_E = np.load(os.path.join(mi_data_dir, f"{subject}_E_data.npy"))
    mi_y_E = np.load(os.path.join(mi_data_dir, f"{subject}_E_label.npy"))

    mi_X = np.concatenate([mi_X_T, mi_X_E], axis=0)
    mi_y = np.concatenate([mi_y_T, mi_y_E], axis=0)

    composite_data, composite_label = [], []

    for _ in range(num_samples):
        # 随机选 MI1, REST, MI2
        mi_idx1, mi_idx2 = random.sample(range(len(mi_X)), 2)
        rest_idx = random.randint(0, len(rest_X) - 1)

        mi_len1 = random.randint(min_len_sec * fs, max_len_sec * fs)
        mi_len2 = random.randint(min_len_sec * fs, max_len_sec * fs)
        rest_len = random.randint(min_len_sec * fs, max_len_sec * fs)

        mi1 = mi_X[mi_idx1][:, :mi_len1]
        mi2 = mi_X[mi_idx2][:, :mi_len2]
        rest = rest_X[rest_idx][:, :rest_len]

        # 滤波接缝：mi1-rest
        trans1 = lowpass_filter(np.concatenate([mi1[:, -50:], rest[:, :50]], axis=1))
        rest[:, :50] = trans1[:, 50:]
        mi1[:, -50:] = trans1[:, :50]

        # 滤波接缝：rest-mi2
        trans2 = lowpass_filter(np.concatenate([rest[:, -50:], mi2[:, :50]], axis=1))
        mi2[:, :50] = trans2[:, 50:]
        rest[:, -50:] = trans2[:, :50]

        merged = np.concatenate([mi1, rest, mi2], axis=1)
        composite_data.append(merged)

        label_dict = {
            "mi1_start": 0,
            "mi1_end": mi_len1,
            "mi1_class": int(mi_y[mi_idx1]),
            "mi2_start": mi_len1 + rest_len,
            "mi2_end": mi_len1 + rest_len + mi_len2,
            "mi2_class": int(mi_y[mi_idx2])
        }
        composite_label.append(label_dict)

    # 保存
    np.save(os.path.join(save_dir, f"{subject}_composite_data.npy"), np.array(composite_data))
    np.save(os.path.join(save_dir, f"{subject}_composite_label.npy"), composite_label)
    print(f"✅ {subject} 生成完成：{len(composite_data)} 个样本")

# === 主执行 ===
for subj in selected_subjects:
    create_samples(subj, num_samples=200)
