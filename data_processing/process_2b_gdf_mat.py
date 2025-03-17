import mne
import scipy.io
import numpy as np
import os

# ----------参数设置----------
gdf_path = './B0101T.gdf'  # GDF文件路径
mat_path = './B0101T.mat'  # MAT标签文件路径
segment_len = 1000  # 每个trial长度（4秒=1000采样点）
rest_len = 250  # 静息段长度（1秒）
save_path = './processed_2b_sample.npz'  # 保存路径

# ----------加载GDF数据----------
print("加载GDF信号...")
raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)  # 使用MNE加载GDF文件
eeg_data = raw.get_data()  # 获取EEG数据，shape: (n_channels, n_samples)
print("EEG数据 shape:", eeg_data.shape)

# ----------加载MAT标签----------
print("加载标签文件...")
mat = scipy.io.loadmat(mat_path)  # 加载.mat文件
trial_starts = mat['trial'].squeeze()  # trial起始位置（单位：采样点）
labels = mat['label'].squeeze()  # 每个trial对应的标签（1=左手, 2=右手）
print("trial数量:", len(trial_starts))

# ----------切片处理----------
X_rest, y_rest = [], []  # 存储静息态数据及标签
X_mi, y_mi = [], []  # 存储运动想象态数据及标签

for i in range(len(trial_starts)):
    start = int(trial_starts[i])  # 当前trial的起始采样点

    # 确保trial长度够segment_len
    if start + segment_len > eeg_data.shape[1]:
        continue

    trial = eeg_data[:, start:start + segment_len]  # 提取一个trial段，shape: (n_channels, 1000)

    # 静息段：cue前1秒（前250采样点）
    rest_seg = trial[:, :rest_len]  # 取前250点作为静息段
    X_rest.append(rest_seg)
    y_rest.append(0)  # 静息态标签设为0

    # 想象段：在trial内随机取1秒（250采样点）
    max_start = segment_len - rest_len  # 随机片段起点的最大值（确保不超出范围）
    rand_start = np.random.randint(0, max_start + 1)  # 随机选一个起点
    mi_seg = trial[:, rand_start:rand_start + rest_len]  # 提取想象段
    X_mi.append(mi_seg)
    y_mi.append(1)  # 想象态标签设为1

# 合并静息态与想象态数据
X = np.array(X_rest + X_mi)  # shape: (样本数, 通道数, 时间点数)
y = np.array(y_rest + y_mi)  # shape: (样本数,)

print("最终数据 shape:", X.shape, y.shape)

# ----------保存处理结果----------
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建保存目录
np.savez(save_path, X=X, y=y)  # 保存为npz文件
print(f"保存成功: {save_path}")
