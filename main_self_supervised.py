import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from data_processing.data_get import get_EEGSet, EEGDataSet, get_EEGSet_BCI_Database
from model_set.net import EEGNet
from experiment import Experiment, weights_init
from model_set.net_compared import ShallowConvNet
from model_set.net_lmda import LMDA
from model_set.conformer import Conformer
from model_set.net_lmda_1921 import LMDA1
from PatchTST_supervised.models import PatchTST
from model_set import iTransformer, MixFormer
from PatchTST_self_supervised.src.models.patchTST import get_model_self
from data_loader import PHYSIONET
import os
import sys
sys.path.append("/home/work3/wkh/CL-Model")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"          # if use '2,1', then in pytorch, gpu2 has id 0

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
        # print('memory allocated:', torch.cuda.memory_allocated() / 1024 ** 2, 'MB')
        # print('max memory allocated:', torch.cuda.max_memory_allocated() / 1024 ** 2, 'MB')
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
        eeg_net = LMDA(num_classes=num_class, chans=channels, samples=samples, classification=model_classification).to(device)
    elif model_name == 'EEG_Conformer':
        # eeg_net = Conformer().to(device)
        eeg_net = Conformer(channel=8, n_classes=4).to(device)
    elif model_name == 'PatchTST':
        args = PatchTST.set_parser()
        eeg_net = PatchTST.PatchTST(args, cla=model_classification).to(device)
    elif model_name == 'iTransformer':
        args = iTransformer.set_parser()
        eeg_net = iTransformer.iTransformer(args, cla=model_classification).to(device)
    elif model_name == 'MixFormer':
        eeg_net = MixFormer.MixFormer(cla=model_classification).to(device)
    elif model_name == 'EEGNet':
        eeg_net = EEGNet(num_classes=4).to(device)
    else:
        eeg_net = ShallowConvNet(num_classes=4, chans=22, samples=1125).to(device)

    # 模型参数初始化
    eeg_net.apply(weights_init)
    return eeg_net

def get_dataloader(batch_size, data_set):
    data_path = "dataset/BCI2a/BCICIV_2a_gdf"
    train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set)
    # train_set, test_set, train_label_set, test_label_set = PHYSIONET.train_test_data(0.8, 0.2)

    # train_set = norm_data(train_set)      # dont need
    # test_set = norm_data(test_set)

    # 定义DataSet
    train_loader = EEGDataSet(train_set, train_label_set)
    test_loader = EEGDataSet(test_set, test_label_set)
    print('训练数据集长度: {}'.format(len(train_loader)))
    print('测试数据集长度: {}'.format(len(test_loader)))

    # DataLoader创建数据集
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

def get_dataloader_BCI_Database(batch_size, data_set):
    data_path = "dataset/BCI Database/Signals/DATA A/"
    # train_set: (80, 27, 1921)    train_label_set: (80, )
    train_set, test_set, train_label_set, test_label_set = get_EEGSet_BCI_Database(data_path, data_set)

    # train_set = norm_data(train_set)
    # test_set = norm_data(test_set)

    # 定义DataSet
    train_loader = EEGDataSet(train_set, train_label_set)
    test_loader = EEGDataSet(test_set, test_label_set)
    print('训练数据集长度: {}'.format(len(train_loader)))
    print('测试数据集长度: {}'.format(len(test_loader)))

    # DataLoader创建数据集
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

def get_data_bci2a(batch_size):
    train_data_phy = np.load('data2a/data_bci2a_train_data.npy')
    train_labels_phy = np.load('data2a/data_bci2a_train_label.npy')
    valid_data_phy = np.load('data2a/data_bci2a_valid_data.npy')
    valid_labels_phy = np.load('data2a/data_bci2a_valid_label.npy')
    test_data_phy = np.load('data2a/data_bci2a_test_data.npy')
    test_labels_phy = np.load('data2a/data_bci2a_test_label.npy')

    test_val = np.concatenate([valid_data_phy, test_data_phy])
    test_val_label = np.concatenate([valid_labels_phy, test_labels_phy])

    train_dataset = EEGDataSet(train_data_phy, train_labels_phy)
    val_dataset = EEGDataSet(valid_data_phy, valid_labels_phy)
    test_dataset = EEGDataSet(test_data_phy, test_labels_phy)
    test_val_dataset = EEGDataSet(test_val, test_val_label)

    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('测试数据集长度: {}'.format(len(test_val_dataset)))

    # DataLoader创建数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

def get_data_bci2a_cross_validation(batch_size, valid_data_set):
    index = 1
    all_data_set = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']        # 
    for data in all_data_set:
        if data == valid_data_set:
            valid_data = np.load('data2a/data_bci2a_' + data + '_data.npy')
            valid_label = np.load('data2a/data_bci2a_' + data + '_label.npy')
        else:
            train_data_set = np.load('data2a/data_bci2a_' + data + '_data.npy')
            train_label_set = np.load('data2a/data_bci2a_' + data + '_label.npy')
            if index == 1:
                all_train_data = train_data_set
                all_train_label = train_label_set
                index += 1
            else:
                all_train_data = np.concatenate((all_train_data, train_data_set), axis=0)
                all_train_label = np.concatenate((all_train_label, all_train_label), axis=0)

    n_random_samples = 2000
    random_indices = np.random.choice(all_train_data.shape[0], size=n_random_samples, replace=False)

    all_train_data = all_train_data[random_indices]
    all_train_label = all_train_label[random_indices]

    train_dataset = EEGDataSet(all_train_data, all_train_label)
    val_dataset = EEGDataSet(valid_data, valid_label)

    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('测试数据集长度: {}'.format(len(val_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    # 随机种子，保证生成的随机数是一样的
    setup_seed(521)  # 521, 322
    print('* * ' * 20)

    # basic info of the dataset
    num_class = 4
    channels = 22
    samples = 1000
    sample_rate = 250

    device = get_device()

    # ===============================  超 参 数 设 置 ================================
    lr_model = 0.001        # 0.0006
    epochs = 100
    batch_size = 64

    temperature = 0.1       # contrastive loss default = 0.1
    # ==============================================================================

    head_type = 'classification'     # 'pretrain'    'classification'       test

    # ============================
    data_set = ['A03']      # 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08'
    # data_set = [f"A{i}" for i in range(1, 30 + 1)]        # BCI_Database
    
    model_name = 'PatchTST_self'     # LMDANet   EEG_Conformer   PatchTST    iTransformer    MixFormer      head_only       PatchTST_self     EEGNet
    model_save_path = "model/" + model_name + "_A01-A07.pth"    # _classification  classification_froze    finetone
    classification = True                                      # False: pre-train contrastive net  or  True: fine-tone
    # ============================

    # train_dataloader, test_dataloader = get_dataloader_BCI_Database(data_set=data_set, batch_size=batch_size)
    # train_dataloader, test_dataloader = get_dataloader(data_set=data_set, batch_size=batch_size)
    # train_dataloader, test_dataloader = get_data_bci2a(batch_size=batch_size)
    train_dataloader, test_dataloader = get_data_bci2a_cross_validation(batch_size=batch_size, valid_data_set='A09')
    
    # model = get_model(model_name=model_name, device=device, model_classification=classification)
    model = get_model_self(c_in=22, head_type=head_type).to(device)     

    # model = LMDA1(num_classes=2, chans=12, samples=1921, channel_depth1=24, channel_depth2=9, classification=True, 
    #               avepool=512//10).cuda().to(device)    
    # model = EEGNet(num_classes=2, chans=12, samples=1921, kernLength=512//2).cuda()
    # model = Conformer(channel=12, n_classes=2, cla=True).to(device)
    # model = get_model_self(c_in=12, head_type=head_type).to(device)

    loss_func = torch.nn.CrossEntropyLoss().to(device)          # 定义交叉熵损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_model, weight_decay=1e-5)       # 权重衰减（L2正则化） weight_decay

    exp = Experiment(model=model,
                     model_name=model_name,
                     optimizer=optimizer,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     device=device,
                     epochs=epochs,
                     model_save_path=model_save_path,
                     temperature=temperature,
                     loss_func=loss_func,
                     classification=classification
                     )

    if head_type == 'pretrain':
        exp.train_run()
    elif head_type == 'classification':
        # exp.run()
        # exp.train_run_finetune()

        exp.run_PatchTST_self(data_aug=True)        # False True
        # exp.run_PatchTST_self_contrast()
        # exp.test_PatchTST_self()

