import os
import numpy as np
import mne
import scipy.io as sio
from scipy.signal import butter, filtfilt

# ---------------- 带通滤波 ----------------
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

# ---------------- 配置路径 ----------------
base_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(base_dir, 'dataset', 'BCI2a', 'BCICIV_2a_gdf')
save_dir = os.path.join(base_dir, 'dataset', 'data_2a_binary_MI_Rest')
os.makedirs(save_dir, exist_ok=True)

subjects = [f"A0{i}" for i in range(1, 10)]
segment_len = 1000  # 4 秒（250Hz 采样率）

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

    for suffix in ['T', 'E']:
        gdf_path = os.path.join(data_dir, f"{subj}{suffix}.gdf")
        mat_path = os.path.join(data_dir, f"{subj}{suffix}.mat")

        if not os.path.exists(gdf_path) or not os.path.exists(mat_path):
            print(f"⚠️ 缺失文件: {gdf_path} 或 {mat_path}")
            continue

        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        raw.pick_channels([ch for ch in target_channels if ch in raw.ch_names])
        fs = int(raw.info['sfreq'])
        eeg_data = raw.get_data()
        eeg_data = apply_bandpass_filter(eeg_data, fs=fs)

        events, event_id = mne.events_from_annotations(raw)

        X_data, y_data = [], []

        if suffix == 'T':
            label_data = sio.loadmat(mat_path)
            labels = label_data.get('classlabel', label_data.get('label')).squeeze() - 1
            cue_events = [v for k, v in event_id.items() if k in ['769', '770', '771', '772']]
            trials = events[np.isin(events[:, 2], cue_events)]

            for i, (start_time, _, _) in enumerate(trials):
                start_time = int(start_time)

                # MI 想象
                mi_start = start_time + int(1 * fs)
                mi_end = mi_start + segment_len
                if mi_end <= eeg_data.shape[1]:
                    seg_mi = eeg_data[:, mi_start:mi_end]
                    if seg_mi.shape[1] == segment_len:
                        X_data.append(seg_mi)
                        y_data.append(1)

                # 静息（cue 前 4 秒）
                rest_start = start_time - segment_len
                rest_end = start_time
                if rest_start >= 0:
                    seg_rest = eeg_data[:, rest_start:rest_end]
                    if seg_rest.shape[1] == segment_len:
                        X_data.append(seg_rest)
                        y_data.append(0)

        else:  # suffix == 'E'
            if '783' not in event_id:
                print(f"⚠️ 无效事件783: {subj}{suffix}")
                continue
            cue_val = event_id['783']
            trials = events[events[:, 2] == cue_val]

            for (start_time, _, _) in trials:
                start_time = int(start_time)

                # MI：cue 后 4 秒
                mi_start = start_time
                mi_end = mi_start + segment_len
                if mi_end <= eeg_data.shape[1]:
                    seg_mi = eeg_data[:, mi_start:mi_end]
                    if seg_mi.shape[1] == segment_len:
                        X_data.append(seg_mi)
                        y_data.append(1)

                # 静息：cue 前 4 秒
                rest_start = start_time - segment_len
                rest_end = start_time
                if rest_start >= 0:
                    seg_rest = eeg_data[:, rest_start:rest_end]
                    if seg_rest.shape[1] == segment_len:
                        X_data.append(seg_rest)
                        y_data.append(0)

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        print(f"✅ 被试 {subj}{suffix} 完成：样本数 = {len(y_data)}")

        np.save(os.path.join(save_dir, f"{subj}_{suffix}_data.npy"), X_data)
        np.save(os.path.join(save_dir, f"{subj}_{suffix}_label.npy"), y_data)

        if suffix == 'T':
            all_X_T.append(X_data)
            all_y_T.append(y_data)
        else:
            all_X_E.append(X_data)
            all_y_E.append(y_data)

# 合并保存所有被试的 T/E 数据
if all_X_T:
    X_T_all = np.concatenate(all_X_T, axis=0)
    y_T_all = np.concatenate(all_y_T, axis=0)
    np.save(os.path.join(save_dir, 'data_2a_T_all.npy'), X_T_all)
    np.save(os.path.join(save_dir, 'label_2a_T_all.npy'), y_T_all)
    print(f"\n📦 所有训练数据合并完成，共 {X_T_all.shape[0]} trials")

if all_X_E:
    X_E_all = np.concatenate(all_X_E, axis=0)
    y_E_all = np.concatenate(all_y_E, axis=0)
    np.save(os.path.join(save_dir, 'data_2a_E_all.npy'), X_E_all)
    np.save(os.path.join(save_dir, 'label_2a_E_all.npy'), y_E_all)
    print(f"📦 所有测试数据合并完成，共 {X_E_all.shape[0]} trials")
