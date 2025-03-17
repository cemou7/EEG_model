import torch
from torchsummary import summary

from model_set.net_lmda import LMDA  # 如果你用的是 LMDA 模型
import numpy as np
from torch.utils.data import DataLoader
from data_processing.data_get import EEGDataSet
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
        model = EEGNet(num_classes=4, chans=22, samples=1000, kernLength=512//2)
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
    print(f"\n✅ 测试集准确率: {acc:.4f} ({correct}/{len(all_labels)} correct)")

    return all_preds, all_labels, acc


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from data_processing.data_get import EEGDataSet

    model_name = 'Conformer' # EEGNet
    model_path = 'model/EEG_Conformer_test_val.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(model_name, model_path, device)


    test_data = np.load('data2a/data_bci2a_test_data.npy')  # 后部分作为测试集
    test_label = np.load('data2a/data_bci2a_test_label.npy')
    test_dataset = EEGDataSet(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    summary(model, input_size=(1, 22, 1000), batch_size=64)

    # 推理+评估准确率
    preds, labels, acc = predict_and_evaluate(model, test_loader, device, model_name)

