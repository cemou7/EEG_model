from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchsummary import summary

from model_set.net_lmda import LMDA  # 如果你用的是 LMDA 模型
import numpy as np
from torch.utils.data import DataLoader
from data_processing.data_get import EEGDataSet, prepare_data
# from model_set.net import EEGNet
# from model_set.net_compared import ShallowConvNet
# from model_set.dfformer import get_dfformer_model
# from model_set.conformer import Conformer
# ... 其他模型 import 按需添加

# ========== 1. 加载模型结构 ==========
def load_model(model_name, model_path, device):
    if model_name == 'LMDANet':
        model = LMDA(num_classes=4, chans=22, samples=1000,
                     channel_depth1=24, channel_depth2=9, kernel=75,
                     depth=9, ave_depth=1, avepool=25, classification=True)
    elif model_name == 'EEGNet':
        from model_set.net import EEGNet
        model = EEGNet(num_classes=2, chans=3, samples=1000, kernLength=512//2)
    elif model_name == 'Conformer':
        from model_set.conformer import Conformer
        model = Conformer(channel=22, n_classes=4, cla=True)
    elif model_name == 'dfformer':
        from model_set.dfformer import get_dfformer_model
        model = get_dfformer_model()
    elif model_name == 'PatchTST':
        from PatchTST_supervised.models import PatchTST
        args = PatchTST.set_parser()
        model = PatchTST.PatchTST(args, cla=True)
    else:
        raise ValueError("Unsupported model name!")

    # 加载权重参数
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 放入设备 & 切换eval模式
    model = model.to(device)
    model.eval()
    return model


def predict_and_evaluate(model, test_dataloader, device, model_name='LMDANet'):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device).float()
            y = y.to(device).long()

            if model_name in ['Conformer', 'LMDANet','EEGNet']:
                X = X.unsqueeze(1)
            elif model_name == 'dfformer':
                X = X.unsqueeze(2)
            elif model_name == 'PatchTST':
                X = X.permute(0, 2, 1)

            output = model(X)
            _, predicted = torch.max(output.data, 1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # ✅ 准确率计算
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    acc = correct / len(all_labels)

    # ✅ 统计每个类别的预测数量
    count_0 = all_preds.count(0)
    count_1 = all_preds.count(1)

    print(f"\n✅ 测试集准确率: {acc:.4f} ({correct}/{len(all_labels)} correct)")
    print(f"📊 预测结果统计：0 类数量 = {count_0}，1 类数量 = {count_1}")

    return all_preds, all_labels, acc

def visualize_tsne(features, labels):
    """
    使用 t-SNE 将高维特征降到 2D，并进行散点图可视化.
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)  # (N, 2)

    plt.figure()
    # c=labels 会根据 label 进行默认配色
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels)
    plt.title("t-SNE Visualization")
    plt.show()

if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from data_processing.data_get import EEGDataSet

    model_name = 'EEGNet' # EEGNet   Conformer 
    model_path = 'model/EEGNet_mi_rest_2b_preproccess.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(model_name, model_path, device)


    channel_names = ['EEG-C3', 'EEG-Cz', 'EEG-C4']
    all_channel_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']  # 替换为你实际的22个通道名
    c3_idx = all_channel_names.index('EEG-C3')
    cz_idx = all_channel_names.index('EEG-Cz')
    c4_idx = all_channel_names.index('EEG-C4')
    channel_indices = [c3_idx, cz_idx, c4_idx]
    # 加载数据
    data = np.load('dataset/data2a_no_reject/A01_data.npy')   # shape: [samples, 22, 1000]
    label = np.load('dataset/data2a_no_reject/A01_label.npy')  # shape: [samples]

    data = prepare_data(data)
    # 取出3个通道的数据作为输入
    data_selected = data[:, channel_indices, :]  # shape: [samples, 3, 1000]



    # test_data = np.load('data_2b_mi_rest/B09_data.npy')  # 后部分作为测试集
    # test_label = np.load('data_2b_mi_rest/B09_label.npy')
    # test_dataset = EEGDataSet(test_data, test_label)
    test_dataset = EEGDataSet(data_selected, label)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    summary(model, input_size=(1, 3, 1000), batch_size=64)

    # 推理+评估准确率
    preds, labels, acc = predict_and_evaluate(model, test_loader, device, model_name)

