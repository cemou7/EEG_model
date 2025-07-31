import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from data_processing.data_get import get_EEGSet, EEGDataSet, prepare_data
from model_set.LMDA_net_GCN import LMDA
# from model_set.lmda_band_graph_conv import LMDA_GCN
from model_set.net import EEGNet
from experiment import Experiment, weights_init
from model_set.net_compared import ShallowConvNet
# from model_set.net_lmda import LMDA
from model_set.conformer import Conformer
from model_set.dfformer import get_dfformer_model
from PatchTST_supervised.models import PatchTST
from model_set import iTransformer, MixFormer
from scipy.signal import coherence, hilbert
from collections import Counter
# from data_loader import PHYSIONET
import os
import sys
sys.path.append("/home/work/CZT/CL-Model")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"          # if use '2,1', then in pytorch, gpu2 has id 0

base_dir = os.path.abspath(os.getcwd())


def setup_seed(seed=521):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('gpu use:' + str(torch.cuda.is_available()))
    print('gpu count: ' + str(torch.cuda.device_count()))

    if torch.cuda.is_available():
        print('device: GPU')
        print('device index:' + str(torch.cuda.current_device()))
        print('memory allocated:', torch.cuda.memory_allocated() / 1024 ** 2, 'MB')
        print('max memory allocated:', torch.cuda.max_memory_allocated() / 1024 ** 2, 'MB')
    else:
        print('device: CPU')

    return device
def norm_data(data):
    mean = np.mean(data)
    std = np.std(data)

    normalized_data = (data - mean) / std
    return normalized_data

def get_model(model_name, device, model_classification,adj_init):
    if model_name == 'LMDANet':
        eeg_net = LMDA(num_classes=num_class, chans=channels, samples=samples,
                       channel_depth1=model_para['channel_depth1'],
                       channel_depth2=model_para['channel_depth2'],
                       kernel=model_para['kernel'], depth=model_para['depth'],
                       ave_depth=model_para['pool_depth'], avepool=model_para['avepool']
                       ,classification=model_classification,adj_init=adj_init
                       )
    elif model_name == 'EEG_Conformer':
        eeg_net = Conformer(channel=22, n_classes=4, cla=True)
        # eeg_net = Conformer(channel=3, n_classes=2, cla=True)
    elif model_name == 'dfformer':
        eeg_net = get_dfformer_model()
    elif model_name == 'PatchTST':
        args = PatchTST.set_parser()
        eeg_net = PatchTST.PatchTST(args, cla=model_classification)
    elif model_name == 'iTransformer':
        args = iTransformer.set_parser()
        eeg_net = iTransformer.iTransformer(args, cla=model_classification)
    elif model_name == 'MixFormer':
        eeg_net = MixFormer.MixFormer(cla=model_classification)
    elif model_name == 'EEGNet':
        eeg_net = EEGNet(num_classes=4, chans=22, samples=1000, kernLength=512//2)
        # eeg_net = EEGNet(num_classes=4, chans=3, samples=1000, kernLength=512//2)
        # eeg_net = EEGNet(num_classes=2, chans=3, samples=1000, kernLength=512//2)
    # elif model_name == 'LMDA_GCN':
    #     eeg_net = LMDA_GCN(num_classes=num_class, chans=channels, samples=samples,
    #                 channel_depth1=model_para['channel_depth1'],
    #                 channel_depth2=model_para['channel_depth2'],
    #                 kernel=model_para['kernel'], depth=model_para['depth'],
    #                 ave_depth=model_para['pool_depth'], avepool=model_para['avepool']
    #                 ,classification=model_classification,adj_init=adj_init
    #                 ) 
    else:
        eeg_net = ShallowConvNet(num_classes=4, chans=22, samples=1125)      
    
        

    # 模型参数初始化
    eeg_net.apply(weights_init)
    eeg_net.to(device)
    return eeg_net
def get_data_bci2a_split_811(subject_id: str, batch_size=64):
    # data_path = f'dataset/data2a/data_bci2a_{subject_id}_data.npy'
    # label_path = f'dataset/data2a/data_bci2a_{subject_id}_label.npy'    
    # data_path = f'dataset/data2a_no_reject/{subject_id}_data.npy'
    # label_path = f'dataset/data2a_no_reject/{subject_id}_label.npy'    
    data_path = f'dataset/data_2a_filter/{subject_id}_data.npy'
    label_path = f'dataset/data_2a_filter/{subject_id}_label.npy'

    data = np.load(data_path)
    labels = np.load(label_path)
    data = prepare_data(data)

    total_samples = len(data)
    train_end = int(total_samples * 0.8)
    val_end = int(total_samples * 0.9)

    train_data, val_data, test_data = data[:train_end], data[train_end:val_end], data[val_end:]
    train_labels, val_labels, test_labels = labels[:train_end], labels[train_end:val_end], labels[val_end:]

    train_dataset = EEGDataSet(train_data, train_labels)
    val_dataset = EEGDataSet(val_data, val_labels)
    test_dataset = EEGDataSet(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[{subject_id}] 训练: {len(train_dataset)} 验证: {len(val_dataset)} 测试: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

def compute_pcc(X):
    return np.corrcoef(X.T)

def compute_coh(X, fs=250):
    C = X.shape[1]
    coh_mat = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            f, Cxy = coherence(X[:, i], X[:, j], fs=fs, nperseg=256)
            coh_mat[i, j] = np.mean(Cxy)
    return coh_mat

def compute_plv(X):
    analytic_signal = hilbert(X, axis=0)
    phase = np.angle(analytic_signal)
    C = X.shape[1]
    plv_mat = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            phase_diff = phase[:, i] - phase[:, j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_mat[i, j] = plv
    return plv_mat

def get_subject_adjacency_matrix(subject_id: str, method='all', fs=250, save_path=None):
    """
    method: 'pcc', 'coh', 'plv', or 'all' (means sum of all three)
    """
    # 加载训练数据
    train_data_path = f'dataset/data_no_reject_TE/{subject_id}_T_data.npy'
    train_data = np.load(train_data_path)  # shape: (N_trials, C, T)

    # 拼接所有 trial 到一个矩阵: [T_total, C]
    N, C, T = train_data.shape
    train_data_2d = train_data.transpose(0, 2, 1).reshape(-1, C)  # [N*T, C]

    # 计算邻接矩阵
    A = None
    if method == 'pcc':
        A = np.abs(compute_pcc(train_data_2d))
    elif method == 'coh':
        A = np.abs(compute_coh(train_data_2d, fs))
    elif method == 'plv':
        A = np.abs(compute_plv(train_data_2d))
    elif method == 'all':
        A = np.abs(compute_pcc(train_data_2d)) + \
            np.abs(compute_coh(train_data_2d, fs)) + \
            np.abs(compute_plv(train_data_2d))
    else:
        raise ValueError("method must be one of ['pcc', 'coh', 'plv', 'all']")

    # 归一化到 [0, 1]
    A = A / A.max()

    # 可选保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, A)

    return A.astype(np.float32)


def get_data_bci2a_TE(subject_id: str, batch_size=64):
    # 使用按 T/E 分离的数据集
  
    # train_data_path = f'dataset/data_no_reject_TE/{subject_id}_T_data.npy'
    # train_label_path = f'dataset/data_no_reject_TE/{subject_id}_T_label.npy'
    # test_data_path = f'dataset/data_no_reject_TE/{subject_id}_E_data.npy'
    # test_label_path = f'dataset/data_no_reject_TE/{subject_id}_E_label.npy'

    data_dir = 'dataset/data_no_reject_TE'
    all_subjects = [f"A0{i}" for i in range(1, 10)]  # 被试 A1 ~ A9

    train_data_list = []
    train_label_list = []

    for subject in all_subjects:
        if subject == subject_id:
            continue  # 留出被试 test_subject_id 用于测试

        # 加入该被试的 T/E 数据（作为训练数据）
        t_data = np.load(os.path.join(data_dir, f"{subject}_T_data.npy"))
        t_label = np.load(os.path.join(data_dir, f"{subject}_T_label.npy"))
        e_data = np.load(os.path.join(data_dir, f"{subject}_E_data.npy"))
        e_label = np.load(os.path.join(data_dir, f"{subject}_E_label.npy"))

        train_data_list.append(t_data)
        train_label_list.append(t_label)
        train_data_list.append(e_data)
        train_label_list.append(e_label)

    # 合并训练集
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_label_list, axis=0)
    test_data_path = f'dataset/data_no_reject_TE/{subject_id}_E_data.npy'
    test_label_path = f'dataset/data_no_reject_TE/{subject_id}_E_label.npy'


    # 加载数据
    # train_data = np.load(train_data_path)
    # train_labels = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)

    # 数据预处理（如标准化、reshape）
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    # 构建 Dataset 和 DataLoader
    train_dataset = EEGDataSet(train_data, train_labels)
    test_dataset = EEGDataSet(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[{subject_id}] 训练: {len(train_dataset)} 测试: {len(test_dataset)}")
    return train_loader, test_loader

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
from torch.utils.data import DataLoader, ConcatDataset

def get_data_bci2a_TE_cv(target_subject: str, batch_size=64):
    """
    交叉验证方式：目标被试用于验证 + 测试，其余被试数据用于训练。
    - 验证集：target_subject_T
    - 测试集：target_subject_E
    - 训练集：其余被试的 T+E
    """

    all_subjects = [f"A0{i}" for i in range(1, 10)]
    assert target_subject in all_subjects, "目标被试 ID 错误"

    train_data_list = []
    val_data, val_label = None, None
    test_data, test_label = None, None

    for subj in all_subjects:
        # t_data = np.load(f'dataset/data_no_reject_TE_1125/{subj}_T_data.npy')
        # t_label = np.load(f'dataset/data_no_reject_TE_1125/{subj}_T_label.npy')
        # e_data = np.load(f'dataset/data_no_reject_TE_1125/{subj}_E_data.npy')
        # e_label = np.load(f'dataset/data_no_reject_TE_1125/{subj}_E_label.npy')        
        t_data = np.load(f'dataset/data_no_reject_TE/{subj}_T_data.npy')
        t_label = np.load(f'dataset/data_no_reject_TE/{subj}_T_label.npy')
        e_data = np.load(f'dataset/data_no_reject_TE/{subj}_E_data.npy')
        e_label = np.load(f'dataset/data_no_reject_TE/{subj}_E_label.npy')

        if subj == target_subject:
            val_data, val_label = t_data, t_label
            test_data, test_label = e_data, e_label
        else:
            train_data_list.append((t_data, t_label))
            train_data_list.append((e_data, e_label))

    # 合并训练数据
    all_train_data = np.concatenate([d for d, l in train_data_list], axis=0)
    all_train_label = np.concatenate([l for d, l in train_data_list], axis=0)

    # 数据预处理
    train_data = prepare_data(all_train_data)
    val_data = prepare_data(val_data)
    test_data = prepare_data(test_data)

    # 构建 Dataset
    train_dataset = EEGDataSet(train_data, all_train_label)
    val_dataset = EEGDataSet(val_data, val_label)
    test_dataset = EEGDataSet(test_data, test_label)

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[{target_subject}] ➤ 训练: {len(train_dataset)}，验证: {len(val_dataset)}，测试: {len(test_dataset)}")
    return train_loader, val_loader, test_loader





def predict_and_evaluate(model, test_dataloader, device, model_name='LMDANet'):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device).float()
            y = y.to(device).long()

            if model_name in ['Conformer', 'LMDANet', 'EEGNet']:
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

    # ✅ 每类统计信息
    pred_counter = Counter(all_preds)
    label_counter = Counter(all_labels)

    print(f"\n✅ 测试集准确率: {acc:.4f} ({correct}/{len(all_labels)} correct)")

    print(f"\n📊 实际标签分布:")
    for k in sorted(label_counter.keys()):
        print(f"   类别 {k} 数量 = {label_counter[k]}")

    print(f"\n📊 预测结果分布:")
    for k in sorted(pred_counter.keys()):
        print(f"   类别 {k} 数量 = {pred_counter[k]}")

    return all_preds, all_labels, acc



if __name__ == '__main__':
    # 随机种子，保证生成的随机数是一样的
    setup_seed(521)  # 521, 322
    print('* * ' * 20)

    # basic info of the dataset
    num_class = 4
    channels = 22
    samples = 1000
    sample_rate = 250

    # ===============================  超 参 数 设 置 ================================
    lr_model = 0.0001
    epochs = 500
    batch_size = 8

    temperature = 0.1       # contrastive loss default = 0.1
    # ==============================================================================

    # LMDA-Net parameter
    model_para = {
        'channel_depth1': 24,  # 推荐时间域的卷积层数比空间域的卷积层数更多
        'channel_depth2': 9,
        'kernel': 75,
        'depth': 9,
        'pool_depth': 1,
        'avepool': sample_rate // 10,  # 还是推荐两步pooling的
        'avgpool_step1': 1,
    }
    kwargs = {'num_workers': 1, 'pin_memory': True}
    # ============================
    subject_ids = [f'A0{i}' for i in range(1, 10)]  # A01 ~ A09
    code = '01'
    for subject_id in subject_ids:
        model_name = 'EEGNet'     # LMDANet   EEG_Conformer   PatchTST    iTransformer    MixFormer   dfformer EEGNet  LMDANet_GCN
        data_name = 'data_2a_TE_eegNet_GCN'    #'2a' 'data2a_no_reject'  'data_2a_TE_GCN'   
        # 构建模型文件夹路径
        model_folder = os.path.join("model", model_name)
        os.makedirs(model_folder, exist_ok=True)

        # 拼接完整的模型保存路径
        model_save_path = os.path.join(model_folder, f"{subject_id}_{data_name}.pth")        
        classification = True                                      # False: pre-train contrastive net  or  True: fine-tone
        # ============================ 

        train_dataloader, test_loader = get_data_bci2a_TE(subject_id=subject_id, batch_size=batch_size)
        # train_dataloader,val_dataloader, test_loader = get_data_bci2a_TE_cv(target_subject=subject_id, batch_size=batch_size)
        # train_dataloader,val_dataloader, test_loader = get_data_bci2a_split_811(subject_id=subject_id, batch_size=batch_size)
        adj_init = get_subject_adjacency_matrix(subject_id=subject_id)
        # adj_init = None
        device = get_device() 
        model = get_model(model_name=model_name, device=device, model_classification=classification,adj_init=adj_init)
        loss_func = torch.nn.CrossEntropyLoss().to(device)          # 定义交叉熵损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_model)

        exp = Experiment(model=model,
                        prune_lambda=1e-4, 
                        model_name=model_name,
                        optimizer=optimizer,
                        
                        train_dataloader=train_dataloader,
                        # val_dataloader=val_dataloader,
                        val_dataloader=None,
                        test_dataloader=test_loader,
                        device=device,
                        epochs=epochs,
                        model_save_path=model_save_path,
                        temperature=temperature,
                        loss_func=loss_func,
                        classification=classification,
                        data_name=data_name,
                        subject=subject_id,
                        code=code
                        )
        # exp.run()
        exp.run_no_val()




