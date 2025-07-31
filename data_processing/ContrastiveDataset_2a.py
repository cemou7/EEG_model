import os
import numpy as np

# 数据路径
data_dir = './dataset/data_2a_binary_MI_Rest'
save_dir = './dataset/contrastive_MI_rest_neg_samples'
os.makedirs(save_dir, exist_ok=True)

# 加载合并后的训练数据
X_all = np.load(os.path.join(data_dir, 'data_2a_T_all.npy'))  # shape: (N, C, T)
y_all = np.load(os.path.join(data_dir, 'label_2a_T_all.npy'))  # shape: (N,)

# 获取正样本（MI = 1）
mi_idx = np.where(y_all == 1)[0]
pos_samples = X_all[mi_idx]

# 构建负样本：从全体数据中随机选两个 trial 取均值（非相邻）
# np.random.seed(2025)
# neg_samples = []
# for _ in range(len(pos_samples)):
#     i, j = np.random.choice(len(X_all), size=2, replace=False)
#     avg_sample = (X_all[i] + X_all[j]) / 2
#     neg_samples.append(avg_sample)
# neg_samples = np.stack(neg_samples)
re_idx = np.where(y_all == 0)[0]
neg_samples = X_all[re_idx]

# 保存正负样本
np.save(os.path.join(save_dir, 'pos_samples.npy'), pos_samples)
np.save(os.path.join(save_dir, 'neg_samples.npy'), neg_samples)

print(f"✅ 正样本数量: {len(pos_samples)}, 负样本数量: {len(neg_samples)}")
print(f"📁 数据保存至: {save_dir}")