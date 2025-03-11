import sys
sys.path.append("/home/work3/wkh/CL-Model")
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PatchTST_self_supervised.src.models.patchTST import get_model_self
torch.manual_seed(42)


def calculate_mse(array1, array2):
    difference = array1 - array2
    squared_difference = np.square(difference)
    mse = np.mean(squared_difference)
    return mse


def mask_random(eeg_signal, mask_radio=0.5):
    """
    Mask approximately 50% of the EEG signal patches randomly.
    """
    # Create a random mask for approximately half of the patches
    random_mask = torch.rand(eeg_signal.size(1)) < mask_radio
    random_mask = random_mask[None, :, None, None].expand_as(eeg_signal)

    # Apply the mask to the EEG signal
    eeg_signal[random_mask] = 0
    return eeg_signal


def find_continuous_segments(y):
    """
    找出连续点的起始位置和终止位置。
    """
    if len(y) < 2:
        return []

    segments = []
    start = 0

    for i in range(1, len(y)):
        if y[i] != y[i - 1] + 1:
            segments.append((start, i - 1))
            start = i
        if i == len(y) - 1:
            segments.append((start, i))
    return segments


data = pd.read_csv('PatchTST_self_supervised/data/ETT/ETTh1.csv')
my = data.values[:, 1:]

# mean = np.mean(my)
# std = np.std(my)
# my = (my - mean) / std
# train_data = my[0:12 * 30 * 24 * 4]

# scale
scaler = StandardScaler()
scaler.fit(my)
my = scaler.transform(my)
print(f"mean: {np.mean(my)}, std: {np.std(my)}")
print(scaler.mean_)

j = 2
x = my[0:1921, j]
x = x.astype(np.float32)

model = get_model_self(c_in=12, head_type='classification').cuda()
# model.load_state_dict(torch.load('saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1000_patch50_stride50_epochs-pretrain50_mask0.5_model1.pth'))
model.load_state_dict(torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1921_patch50_stride50_epochs-pretrain100_mask0.5_model1_context1921.pth'))

x = torch.tensor(x).cuda().float()
x = torch.reshape(x, [1, 38, 1, 50])
origin_x = x.clone()
x = mask_random(x, mask_radio=0.5)
outputs = model(x)

x = torch.flatten(x)
origin_x = torch.flatten(origin_x)
outputs = torch.flatten(outputs)
x = x.cpu().squeeze().detach().numpy()
origin_x = origin_x.cpu().squeeze().detach().numpy()
outputs = outputs.cpu().squeeze().detach().numpy()
print(f"MSE: {calculate_mse(origin_x, outputs)}")

x_coords_x = np.arange(0, 1921)

plt.figure(figsize=(20, 10))
x_filtered = x_coords_x[x == 0]
y_filtered = x[x == 0]      # 找到被mask的下标
y_pred = outputs[x == 0]    # 找到mask下标对应的y值

# rescale
origin_x = origin_x * scaler.scale_[j] + scaler.mean_[j]
y_pred = y_pred * scaler.scale_[j] + scaler.mean_[j]

plt.plot(x_coords_x, origin_x, color='blue', label='original data')
# plt.plot(x_filtered, y_pred, color='red', linestyle='-', label='predicted data')

segments = find_continuous_segments(x_filtered)
for i, (start, end) in enumerate(segments):
    if i == 0:
        # 只在第一段线段中添加标签
        plt.plot(x_filtered[start:end+1], y_pred[start:end+1], color='red', label='predicted data')
    else:
        plt.plot(x_filtered[start:end+1], y_pred[start:end+1], color='red')

plt.legend()
plt.show()
# plt.savefig('pred_finetone.png')
