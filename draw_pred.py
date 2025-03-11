import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PatchTST_supervised.models.PatchTST import set_parser, PatchTST
from data_processing.data_get import get_EEGSet

# data = pd.read_csv('./data/ETT/ETTh1.csv')
# my = data.values[:, 1:]

data_path = "dataset/BCI2a/BCICIV_2a_gdf"
data_set = ['A03']
train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set)

mean = np.mean(train_set)
std = np.std(train_set)
train_set = (train_set - mean) / std
mean = np.mean(test_set)
std = np.std(test_set)
test_set = (test_set - mean) / std

my = train_set[7]
print(train_label_set[7])

x = my[1, 0:1000]
x_left = x[0:500]
x_right = x[500:]
x = np.array(x)

args = set_parser()
model = PatchTST(args).cuda()
model.load_state_dict(torch.load('model/PatchTST_A03_checkpoint.pth'))

x_right = torch.tensor(x_right).cuda().float()
x_right = torch.reshape(x_right, [1, 500, 1])
outputs = model(x_right)
outputs = outputs.cpu().squeeze().detach().numpy()

# out = np.concatenate([x_left, outputs])
# plt.plot(out)
# plt.show()

x_coords_x = np.arange(0, 1000)
x_coords_right = np.arange(500, 1000)

plt.figure(figsize=(20, 10))
plt.plot(x_coords_x, x, color='blue', label='original data')
plt.plot(x_coords_right, outputs, color='red', label='predicted data')

plt.legend()
plt.savefig('pred_finetone.png')
