import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from data_processing.data_get import get_EEGSet, EEGDataSet
from model_set.net import EEGNet
from experiment import Experiment, weights_init
from model_set.net_compared import ShallowConvNet
from model_set.net_lmda import LMDA
from model_set.conformer import Conformer
from model_set.dfformer import get_dfformer_model
from PatchTST_supervised.models import PatchTST
from model_set import iTransformer, MixFormer
from data_loader import PHYSIONET
import os
import sys
sys.path.append("/home/work3/wkh/CL-Model")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"          # if use '2,1', then in pytorch, gpu2 has id 0

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


def get_dataloader_save(batch_size, data_set):
    data_path = "dataset\BCICIV_2a_gdf"
    # train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set)
    #
    # # 存储数据
    # np.save('dataset\\train_set_A01-A08.npy', train_set)
    # np.save('dataset\\test_set_A01-A08.npy', test_set)
    # np.save('dataset\\train_label_set_A01-A08.npy', train_label_set)
    # np.save('dataset\\test_label_set_A01-A08.npy', test_label_set)

    # 读取数据
    train_set = np.load('dataset\\train_set_A01-A08.npy')
    test_set = np.load('dataset\\test_set_A01-A08.npy')
    train_label_set = np.load('dataset\\train_label_set_A01-A08.npy')
    test_label_set = np.load('dataset\\test_label_set_A01-A08.npy')

    # 定义DataSet
    train_loader = EEGDataSet(train_set, train_label_set)
    test_loader = EEGDataSet(test_set, test_label_set)
    print('训练数据集长度: {}'.format(len(train_loader)))
    print('测试数据集长度: {}'.format(len(test_loader)))

    # DataLoader创建数据集
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

def norm_data(data):
    mean = np.mean(data)
    std = np.std(data)

    normalized_data = (data - mean) / std
    return normalized_data


def get_dataloader(batch_size, data_set):
    data_path = "dataset/BCI2a/BCICIV_2a_gdf"

    # label  0: Left hand, 1: Right hand, 2: Both feet, 3: Tongue
    train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set)
    # train_set, test_set, train_label_set, test_label_set = PHYSIONET.train_test_data(0.8, 0.2)

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


def get_pre_dataloader(batch_size, data_set):
    data_path = np.load("data2a/data_bci2a_A07_data.npy")
    label_path = np.load("data2a/data_bci2a_A07_label.npy")

    train_set, test_set = data_path[0:288], data_path[288:]
    train_label_set, test_label_set = label_path[0:288], label_path[288:]

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


def get_model(model_name, device, model_classification):
    if model_name == 'LMDANet':
        eeg_net = LMDA(num_classes=num_class, chans=channels, samples=samples,
                       channel_depth1=model_para['channel_depth1'],
                       channel_depth2=model_para['channel_depth2'],
                       kernel=model_para['kernel'], depth=model_para['depth'],
                       ave_depth=model_para['pool_depth'], avepool=model_para['avepool'],
                       classification=model_classification
                       )
    elif model_name == 'EEG_Conformer':
        # eeg_net = Conformer()
        eeg_net = Conformer(channel=22, n_classes=4, cla=True)
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
        eeg_net = EEGNet(num_classes=4)
    else:
        eeg_net = ShallowConvNet(num_classes=4, chans=22, samples=1125)

    # 模型参数初始化
    eeg_net.apply(weights_init)
    eeg_net.to(device)
    return eeg_net


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
                all_train_label = np.concatenate((all_train_label, train_label_set), axis=0)

    # n_random_samples = 2000
    # random_indices = np.random.choice(all_train_data.shape[0], size=n_random_samples, replace=False)

    # all_train_data = all_train_data[random_indices]
    # all_train_label = all_train_label[random_indices]

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

    # ===============================  超 参 数 设 置 ================================
    lr_model = 0.003
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
    data_set = 'A01'      # 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08'
    model_name = 'LMDANet'     # LMDANet   EEG_Conformer   PatchTST    iTransformer    MixFormer   dfformer
    model_save_path = "model/" + model_name + "_valid_A09.pth"    # _classification  _diff_CL_64
    classification = True                                      # False: pre-train contrastive net  or  True: fine-tone
    # ============================

    # train_dataloader, test_dataloader = get_dataloader(data_set=data_set, batch_size=batch_size)
    # train_dataloader, test_dataloader = get_pre_dataloader(data_set=data_set, batch_size=batch_size)
    train_dataloader, test_dataloader = get_data_bci2a_cross_validation(batch_size=batch_size, valid_data_set=data_set)

    device = get_device()
    model = get_model(model_name=model_name, device=device, model_classification=classification)
    loss_func = torch.nn.CrossEntropyLoss().to(device)          # 定义交叉熵损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_model)

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

    # if not classification:
    #     # 1. backbone pre-train
    #     print("start: backbone pre-train")
    #     exp.train_run()
    # else:
    #     # 2. fine-tune
    #     print("start: model fine-tune")
    #     exp.train_run_finetune()

    # 3. draw features t-SNE UMAP
    # exp.draw_feature(method='t-SNE', test_draw=False, point_num=288)

    # draw_point_feature
    # train_dataloader_A09, test_dataloader_A09 = get_dataloader(data_set=['A09'], batch_size=batch_size)
    # exp.draw_point_feature(train_data=test_dataloader_A09)

    exp.run()
    # exp.train_run()

    # exp.run_add_loss_contrast(lamda=0.6)
    # exp.test_run_classification()

    # exp.draw_feature(test_draw=True, point_num=288, method='UMAP')     # t-SNE  UMAP
