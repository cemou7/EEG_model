import os
import numpy as np
import mne
import scipy.io as sio
from scipy.signal import butter, filtfilt


# 带通滤波函数（4-38 Hz）
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut=4, highcut=38, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered

# 配置路径
base_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(base_dir, 'dataset', 'BCI2a', 'BCICIV_2a_gdf')
# save_dir = os.path.join(base_dir, 'dataset/data_no_reject_TE')
save_dir = os.path.join(base_dir, 'dataset/data_2a_TE_no_reject')
os.makedirs(save_dir, exist_ok=True)

segment_len = 1000  # 4秒 = 1000 sample
subjects = [f"A0{i}" for i in range(1, 10)]
target_channels = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
    'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'
]

# 合并数据的容器
all_X_T, all_y_T = [], []
all_X_E, all_y_E = [], []

for subj in subjects:
    print(f"\n🧠 处理被试 {subj}...")
    X_T, y_T, X_E, y_E = [], [], [], []

    for suffix in ['T', 'E']:
        gdf_file = f"{subj}{suffix}.gdf"
        mat_file = f"{subj}{suffix}.mat"
        gdf_path = os.path.join(data_dir, gdf_file)
        mat_path = os.path.join(data_dir, mat_file)

        if not os.path.exists(gdf_path) or not os.path.exists(mat_path):
            print(f"⚠️ 缺失文件: {gdf_file} 或对应标签")
            continue

        # 加载 EEG 数据
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        raw.pick_channels([ch for ch in target_channels if ch in raw.ch_names])
        eeg_data = raw.get_data()
        eeg_data = apply_bandpass_filter(eeg_data, fs=int(raw.info['sfreq']))

        events, event_id = mne.events_from_annotations(raw)

        # 加载标签
        label_data = sio.loadmat(mat_path)
        labels = label_data.get('classlabel', label_data.get('label')).squeeze() - 1  # 从0开始
        label_idx = 0

        if suffix == 'E':
            trial_starts = events[events[:, 2] == event_id.get('783', -1), 0]
        else:
            trial_starts = events[np.isin(events[:, 2], [event_id.get(str(k), -1) for k in range(769, 773)]), 0]

        for start in trial_starts:
            start = int(start)
            trial_window = events[(events[:, 0] >= start) & (events[:, 0] < start + 1500)]

            # # 跳过包含异常事件的 trial
            # if any(e[2] == event_id.get('1023', -999) for e in trial_window):
            #     label_idx += 1
            #     continue

            end = start + segment_len
            if end > eeg_data.shape[1]:
                label_idx += 1
                continue

            seg = eeg_data[:, start:end]
            if seg.shape[1] != segment_len:
                label_idx += 1
                continue

            label = labels[label_idx] if label_idx < len(labels) else 0
            if suffix == 'T':
                X_T.append(seg)
                y_T.append(label)
            else:
                X_E.append(seg)
                y_E.append(label)

            label_idx += 1

    # 保存被试 T/E 数据
    if X_T:
        np.save(os.path.join(save_dir, f'{subj}_T_data.npy'), np.array(X_T))
        np.save(os.path.join(save_dir, f'{subj}_T_label.npy'), np.array(y_T))
        print(f"✅ 保存: {subj}_T_data.npy ({len(X_T)} trials)")
        all_X_T.append(np.array(X_T))
        all_y_T.append(np.array(y_T))

    if X_E:
        np.save(os.path.join(save_dir, f'{subj}_E_data.npy'), np.array(X_E))
        np.save(os.path.join(save_dir, f'{subj}_E_label.npy'), np.array(y_E))
        print(f"✅ 保存: {subj}_E_data.npy ({len(X_E)} trials)")
        all_X_E.append(np.array(X_E))
        all_y_E.append(np.array(y_E))

# 合并保存所有被试的 T/E 数据
if all_X_T:
    np.save(os.path.join(save_dir, 'data_2a_T_all.npy'), np.concatenate(all_X_T, axis=0))
    np.save(os.path.join(save_dir, 'label_2a_T_all.npy'), np.concatenate(all_y_T, axis=0))
    print(f"\n📦 所有训练数据合并完成，共 {sum(x.shape[0] for x in all_X_T)} trials")

if all_X_E:
    np.save(os.path.join(save_dir, 'data_2a_E_all.npy'), np.concatenate(all_X_E, axis=0))
    np.save(os.path.join(save_dir, 'label_2a_E_all.npy'), np.concatenate(all_y_E, axis=0))
    print(f"📦 所有测试数据合并完成，共 {sum(x.shape[0] for x in all_X_E)} trials")
