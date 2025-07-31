import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from data_processing.data_get import get_EEGSet, EEGDataSet, prepare_data
# from model_set.LMDA_net_GCN import LMDA
# from model_set.lmda_band_graph_conv import LMDA_GCN
from model_set.EEGNet_ATTEN_Multi_Task import EEGNet_ATTEN_Multi_Task
from model_set.LMAD import LMDA
from model_set.LMAD_Multi_Task import LMDA_Multi_Task
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
from torch.utils.data import Dataset, TensorDataset
import os
import sys
sys.path.append("/home/work/CZT/CL-Model")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"          # if use '2,1', then in pytorch, gpu2 has id 

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

def get_model(model_name, device, model_classification):
    if model_name == 'LMDANet':
        eeg_net = LMDA(num_classes=num_class, chans=channels, samples=samples,
                       channel_depth1=model_para['channel_depth1'],
                       channel_depth2=model_para['channel_depth2'],
                       kernel=model_para['kernel'], depth=model_para['depth'],
                       ave_depth=model_para['pool_depth'], avepool=model_para['avepool']
                    #    ,classification=model_classification
                       )    
    elif model_name == 'LMDA_Multi_Task':
        eeg_net = LMDA_Multi_Task(num_classes1=2,num_classes2=4, chans=channels, samples=samples,
                       channel_depth1=model_para['channel_depth1'],
                       channel_depth2=model_para['channel_depth2'],
                       kernel=model_para['kernel'], depth=model_para['depth'],
                       ave_depth=model_para['pool_depth'], avepool=model_para['avepool']
                       )
    elif model_name == 'EEGNet_ATTEN_Multi_Task':
        eeg_net = EEGNet_ATTEN_Multi_Task(Chans=22,kernLength1=36,kernLength2=24,kernLength3=18,
                                          F1=16,D=2,num_classes1=2,num_classes2=4,DOR=0.5)
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

    data_dir = 'dataset/data_2a_TE_no_reject'  #data_2a_TE_no_reject   data_2a_TE   data_2a_TE_all

    # # 当前被试自己的训练和测试数据
    train_data_path = os.path.join(data_dir, f"{subject_id}_T_data.npy")
    train_label_path = os.path.join(data_dir, f"{subject_id}_T_label.npy")
    test_data_path = os.path.join(data_dir, f"{subject_id}_E_data.npy")
    test_label_path = os.path.join(data_dir, f"{subject_id}_E_label.npy")    
    # 所有的训练和测试数据
    # train_data_path = os.path.join(data_dir, "data_2a_T_all.npy")
    # train_label_path = os.path.join(data_dir, "label_2a_T_all.npy")
    # test_data_path = os.path.join(data_dir, "data_2a_E_all.npy")
    # test_label_path = os.path.join(data_dir, "label_2a_E_all.npy")

    # 加载数据
    train_data = np.load(train_data_path)
    train_labels = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)

    # 数据预处理（标准化、reshape）
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    # 构建 Dataset 和 DataLoader
    train_dataset = EEGDataSet(train_data, train_labels)
    test_dataset = EEGDataSet(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[{subject_id}] 训练: {len(train_dataset)} 测试: {len(test_dataset)}")
    return train_loader, test_loader
def get_data_bci2a_Mi_Rest(batch_size=64):
    # 使用按 T/E 分离的数据集

    data_dir = 'dataset/data_2a_binary_MI_Rest'  #data_2a_TE_no_reject   data_2a_TE   data_2a_TE_all  data_2a_binary_MI_Rest

    # 当前被试自己的训练和测试数据
    train_data_path = os.path.join(data_dir, "data_2a_T_all.npy")
    train_label_path = os.path.join(data_dir, "label_2a_T_all.npy")
    test_data_path = os.path.join(data_dir, "data_2a_E_all.npy")
    test_label_path = os.path.join(data_dir, "label_2a_E_all.npy")

    # 加载数据
    train_data = np.load(train_data_path)
    train_labels = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)

    # 数据预处理（标准化、reshape）
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    # 构建 Dataset 和 DataLoader
    train_dataset = EEGDataSet(train_data, train_labels)
    test_dataset = EEGDataSet(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[{subject_id}] 训练: {len(train_dataset)} 测试: {len(test_dataset)}")
    return train_loader, test_loader

def get_contrastive_dataloaders(batch_size=64):
    """
    从指定目录加载正负样本，并返回两个 DataLoader。
    
    参数:
        data_path (str): 存储 pos_samples.npy 和 neg_samples.npy 的路径
        batch_size (int): DataLoader 的 batch 大小
    
    返回:
        pos_loader, neg_loader (DataLoader): 正样本和负样本的 DataLoader
    """
    # data_path = 'dataset/contrastive_MI'
    data_path = 'dataset/contrastive_MI_rest_neg_samples'
    data_dir = 'dataset/data_2a_binary_MI_Rest'

    pos_samples = np.load(os.path.join(data_path, 'pos_samples.npy'))  # shape: [N, C, T]
    neg_samples = np.load(os.path.join(data_path, 'neg_samples.npy'))  # shape: [N, C, T]
    test_data_path = os.path.join(data_dir, "data_2a_E_all.npy")
    test_label_path = os.path.join(data_dir, "label_2a_E_all.npy")
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)
    test_data = prepare_data(test_data)
    test_dataset = EEGDataSet(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 转为 TensorDataset，不关心 label（仅需样本）
    pos_dataset = TensorDataset(torch.tensor(pos_samples, dtype=torch.float32), torch.zeros(len(pos_samples)))
    neg_dataset = TensorDataset(torch.tensor(neg_samples, dtype=torch.float32), torch.zeros(len(neg_samples)))

    pos_loader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    neg_loader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return pos_loader, neg_loader,test_loader
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
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
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

    #  准确率计算
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    acc = correct / len(all_labels)

    #  每类统计信息
    pred_counter = Counter(all_preds)
    label_counter = Counter(all_labels)

    print(f"\n 测试集准确率: {acc:.4f} ({correct}/{len(all_labels)} correct)")

    print(f"\n 实际标签分布:")
    for k in sorted(label_counter.keys()):
        print(f"   类别 {k} 数量 = {label_counter[k]}")

    print(f"\n 预测结果分布:")
    for k in sorted(pred_counter.keys()):
        print(f"   类别 {k} 数量 = {pred_counter[k]}")

    return all_preds, all_labels, acc



if __name__ == '__main__':
    # 随机种子，保证生成的随机数是一样的
    setup_seed(2025)  # 521, 322
    print('* * ' * 20)

    # basic info of the dataset
    num_class = 2
    channels = 22
    samples = 1000
    sample_rate = 250

    # ===============================  超 参 数 设 置 ================================
    lr_model = 0.002
    epochs = 300
    batch_size = 64

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
    # Lmda best parameter batch=16  lr_model=0.002
    subject_ids = [f'A0{i}' for i in range(1, 10)]  # A01 ~ A09
    code = '12'
    all_test_acc = []
    model_name = 'EEGNet_ATTEN_Multi_Task'     # LMDANet  LMDA_Multi_Task EEGNet_ATTEN_Multi_Task
    data_name = 'data_2a_EEGNet_ATTEN_Multi_Task_Classification_with_ResidualMLP_batch64'    #'2a' 'data2a_no_reject'  'data_2a_TE_GCN'   
    summary_dir = f"./logs/result/{model_name}/{code}_{data_name}.txt"
    for subject_id in subject_ids:
    # if True:
    #     subject_id = all
        # 构建模型文件夹路径
        model_folder = os.path.join("model", model_name)
        os.makedirs(model_folder, exist_ok=True)

        # 拼接完整的模型保存路径
        model_save_path = os.path.join(model_folder, f"{subject_id}_{data_name}.pth")        
        classification = True                                      # False: pre-train contrastive net  or  True: fine-tone
        # ============================ 
        train_dataloader, test_loader = get_data_bci2a_Mi_Rest(batch_size=batch_size)# 预训练数据
        train_dataloader_task2, test_dataloader_task2  = get_data_bci2a_TE(subject_id=subject_id, batch_size=batch_size)# 分类数据
        device = get_device() 
        model = get_model(model_name=model_name, device=device, model_classification=classification)
        loss_func = torch.nn.CrossEntropyLoss().to(device)          # 定义交叉熵损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_model)

        exp = Experiment(model=model,
                        prune_lambda=1e-4,  
                        model_name=model_name,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader,
                        train_dataloader_task2 = train_dataloader_task2,
                        val_dataloader=None,
                        num_classes=num_class,
                        test_dataloader=test_loader,
                        test_dataloader_task2 = test_dataloader_task2,
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
        _, _,_,best_test_acc = exp.run_Multi_Task()
        all_test_acc.append(best_test_acc)
    with open(summary_dir, "a") as f:
        f.write("--------------------------------------------------\n")
        f.write(f"Average Test Accuracy across subjects on Supervised: {sum(all_test_acc) / len(all_test_acc):.4f}\n")
        f.write("==================================================\n")
