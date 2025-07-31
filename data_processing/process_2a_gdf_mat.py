import os
import numpy as np
import mne
import scipy.io as sio

# 设置路径
base_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(base_dir, 'dataset', 'BCI2a', 'BCICIV_2a_gdf')
label_dir = os.path.join(base_dir, 'dataset', 'BCI2a', 'BCICIV_2a_gdf')
save_dir = os.path.join(base_dir, 'data2a_filter')
os.makedirs(save_dir, exist_ok=True)

segment_len = 1000  # 4秒 = 1000 sample（250Hz）
sampling_rate = 250
subjects = [f"A0{i}" for i in range(1, 10)]

all_X, all_y = [], []

for subj in subjects:
    X_all, y_all = [], []
    print(f"\n🧠 处理被试 {subj}...")

    for suffix in ['T', 'E']:
        gdf_file = f"{subj}{suffix}.gdf"
        gdf_path = os.path.join(data_dir, gdf_file)
        mat_file = os.path.join(label_dir, f"{subj}{suffix}.mat")

        if not os.path.exists(gdf_path) or not os.path.exists(mat_file):
            print(f"⚠️ 缺失文件: {gdf_file} 或对应标签")
            continue

        # 读取数据
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        # picks = [raw.ch_names.index(ch) for ch in ['EEG-C3', 'EEG-Cz', 'EEG-C4']]
        raw.pick_channels(raw.ch_names[:22])
        eeg_data = raw.get_data()
        events, event_id = mne.events_from_annotations(raw)

        # 加载标签
        label_data = sio.loadmat(mat_file)
        labels = label_data['classlabel'].squeeze()
        labels = labels - 1  # 标签编号从0开始
        label_idx = 0

        if suffix == 'E':
            trial_starts = events[events[:, 2] == event_id.get('783', -1), 0]
        else:
            trial_starts = events[np.isin(events[:, 2], [event_id.get(str(k), -1) for k in range(769, 773)]), 0]

        for start in trial_starts:
            trial_window = events[(events[:, 0] >= start) & (events[:, 0] < start + 1500)]
            if any(e[2] == event_id.get('1023', -999) for e in trial_window):
                label_idx += 1  # 跳过标签
                continue

            start, end = int(start), int(start) + segment_len
            if end > eeg_data.shape[1]:
                label_idx += 1
                continue

            seg = eeg_data[:, start:end]
            if seg.shape[1] == segment_len:
                X_all.append(seg)
                if label_idx < len(labels):
                    y_all.append(labels[label_idx])
                else:
                    y_all.append(0)  # fallback
            label_idx += 1

    # 保存被试数据
    if X_all:
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        np.save(os.path.join(save_dir, f'{subj}_data.npy'), X_all)
        np.save(os.path.join(save_dir, f'{subj}_label.npy'), y_all)
        print(f"✅ 保存 {subj} 完成: {X_all.shape[0]} 条样本")
        all_X.append(X_all)
        all_y.append(y_all)

# 合并所有数据
if all_X:
    merged_X = np.concatenate(all_X, axis=0)
    merged_y = np.concatenate(all_y, axis=0)
    np.save(os.path.join(save_dir, 'data_2a_all.npy'), merged_X)
    np.save(os.path.join(save_dir, 'label_2a_all.npy'), merged_y)
    print(f"\n✅ 所有被试合并完成: 共 {merged_X.shape[0]} 样本")
