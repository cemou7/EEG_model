import sys
sys.path.append("/home/work3/wkh/CL-Model")
from scipy.signal import firwin, lfilter, filtfilt
from scipy.linalg import sqrtm
import mne
from data_processing.data_extract import BCICompetition4Set2A, extract_segment_trial
import os
import numpy as np
from torch.utils.data import Dataset, TensorDataset
import torch
from scipy.signal import butter, filtfilt
class EEGDataSet(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)  # label without one-hot coding

    def __getitem__(self, idx):
        data_tensor = self.data[idx]
        label_tensor = self.labels[idx]
        return data_tensor, label_tensor

    def __len__(self):
        return len(self.data)

    def get_label(self):
        return self.labels


def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.

    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)


def bandpass_cnt(data, low_cut_hz, high_cut_hz, fs, filt_order=200, zero_phase=False):
    # firwin()函数生成一个带通滤波器的窗函数（window）
    win = firwin(filt_order, [low_cut_hz, high_cut_hz], window='blackman', fs=fs, pass_zero='bandpass')

    # 使用lfilter()线性滤波器函数将输入数据data与滤波器系数进行卷积，从而对数据进行带通滤波。
    data_bandpassed = lfilter(win, 1, data)

    # 如果参数zero_phase为True，函数将使用filtfilt()函数对数据进行零相位滤波。
    if zero_phase:
        data_bandpassed = filtfilt(win, 1, data)

    # 函数返回经过带通滤波后的数据data_bandpassed
    return data_bandpassed


# 1. 对每个样本的所有通道的数据进行独立归一化。保证通道每个样本是独立的
def data_norm(data):
    """
    对数据进行归一化 [-1, 1]
    :param data:   ndarray ,shape[N,channel,samples]
    :return:
    """
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.max(abs(data[i]))

    return data_copy

# 2. 对整个数据集进行归一化，计算所有样本的整体平均值和标准差。EEG数据需要被整体分析
def norm_data(data):
    mean = np.mean(data)
    std = np.std(data)
    torch.float32
    normalized_data = (data - mean) / std
    return normalized_data

# 计算输入数据的平均协方差矩阵，然后对每个数据样本进行白化操作，以减少数据样本之间的相关性。
def preprocess_ea(data):
    R_bar = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)
    R_bar_mean = R_bar / len(data)

    for i in range(len(data)):
        data[i] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), data[i])
    return data


def prepare_data(data):
    data_preprocss = data_norm(data)
    data_ea = preprocess_ea(data_preprocss)
    return data_ea


def get_EEGSet(data_path, data_set):
    train_set = list()
    test_set = list()
    train_label_set = list()
    test_label_set = list()

    index = 1
    for subject_Id in data_set:  # ['A01']
        train_filename = "{}T.gdf".format(subject_Id)
        test_filename = "{}E.gdf".format(subject_Id)
        train_filepath = os.path.join(data_path, train_filename)
        test_filepath = os.path.join(data_path, test_filename)
        train_label_filepath = train_filepath.replace(".gdf", ".mat")
        test_label_filepath = test_filepath.replace(".gdf", ".mat")

        train_loader = BCICompetition4Set2A(
            train_filepath, labels_filename=train_label_filepath
        )
        test_loader = BCICompetition4Set2A(
            test_filepath, labels_filename=test_label_filepath
        )
        train_cnt = train_loader.load()
        test_cnt = test_loader.load()

        # 带通滤波
        train_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                     filt_order=200, fs=250, zero_phase=False), train_cnt)

        test_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                    filt_order=200, fs=250, zero_phase=False), test_cnt)

        # train_data = {ndarray:(288, 22, 1125)}
        train_data, train_label = extract_segment_trial(train_cnt)
        test_data, test_label = extract_segment_trial(test_cnt)

        # train_label = ndarray:(288, )
        train_label = train_label - 1
        test_label = test_label - 1

        # preprocessed_train = ndarray:(288, 22, 1125)
        train_data = prepare_data(train_data)
        test_data = prepare_data(test_data)

        if index == 1:
            train_set = train_data
            test_set = test_data
            train_label_set = train_label
            test_label_set = test_label
            index += 1
        else:
            train_set = np.concatenate((train_set, train_data), axis=0)
            test_set = np.concatenate((test_set, test_data), axis=0)
            train_label_set = np.concatenate((train_label_set, train_label), axis=0)
            test_label_set = np.concatenate((test_label_set, test_label), axis=0)

    return train_set, test_set, train_label_set, test_label_set


# -------------------------------------------------------------------------------------------------------------
def read_file(filename):
    raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR',
                              exclude=(['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C5',
                                        'C6', 'F4', 'FC6', 'CP6', 'P4', 'F3', 'FC5', 'CP5', 'P3']))
    raw.load_data()
    data = raw.get_data()
    # print(filename)

    for i_chan in range(data.shape[0]):  # 遍历 channel
        # 将数组中的所有值设置为nan，然后将这些NaN值替换为该数组的均值。
        this_chan = data[i_chan]
        data[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(data[i_chan])
        chan_mean = np.nanmean(data[i_chan])
        data[i_chan, mask] = chan_mean

    # 获取事件和事件 ID
    events, events_id = mne.events_from_annotations(raw)
    MI_L = events_id['769']
    MI_R = events_id['770']

    # 利用mne.io.RawArray类重新创建Raw对象，已经没有nan数据了
    raw_gdf_ = mne.io.RawArray(data, raw.info, verbose="ERROR")

    # 4-38 hz 带通滤波  filt_order:滤波器的阶数
    # raw_gdf = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
    #                 filt_order=100, fs=512, zero_phase=False), raw_gdf_)

    # 使用 resample 方法降采样 512hz -> 250hz
    # raw_gdf = raw_gdf_.copy().resample(250, npad='auto')
    # 使用 filter 带通滤波
    raw_gdf = raw_gdf_.copy()
    raw_gdf.filter(l_freq=4.0, h_freq=30.0, fir_design='firwin')

    # 选择范围为 Cue 后 1.25s - 5s 的数据
    tmin, tmax = 1.25, 5.
    event_id = dict({'769': MI_L, '770': MI_R})

    epochs = mne.Epochs(raw_gdf, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)

    # ndarray.  切片，获取 events 的最后一列
    labels = epochs.events[:, -1] - MI_L
    # ndarray.  Get all epochs as a 3D array.
    data = epochs.get_data()

    return data, labels


def read_modal_file(filename):
    raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR')
    raw.load_data()
    EEG_data = raw.get_data(picks=['C1', 'C3', 'C2', 'C4', 'FC2', 'FC4', 'CP2', 'CP4', 'FC1', 'FC3', 'CP1', 'CP3', 'EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'])

    # 获取事件和事件 ID
    events, events_id = mne.events_from_annotations(raw)
    MI_L = events_id['769']
    MI_R = events_id['770']

    # 使用 filter 带通滤波
    EEG_data_gdf = EEG_data.copy()
    EEG_data_gdf.filter(l_freq=4.0, h_freq=30.0, fir_design='firwin', picks=[])

    # 选择范围为 Cue 后 1.25s - 5s 的数据
    tmin, tmax = 1.25, 5.
    event_id = dict({'769': MI_L, '770': MI_R})

    epochs = mne.Epochs(EEG_data_gdf, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)

    # ndarray.  切片，获取 events 的最后一列
    labels = epochs.events[:, -1] - MI_L
    # ndarray.  Get all epochs as a 3D array.
    data = epochs.get_data()

    return data, labels


def get_EEGSet_BCI_Database(data_path, data_set):
    train_set = list()
    test_set = list()
    train_label_set = list()
    test_label_set = list()

    index = 1
    for subject_Id in data_set:  # ['A2']
        train_filename_1_ = "{}/{}_R1_acquisition.gdf".format(subject_Id, subject_Id)
        train_filename_2_ = "{}/{}_R2_acquisition.gdf".format(subject_Id, subject_Id)
        test_filename = "{}/{}_R3_onlineT.gdf".format(subject_Id, subject_Id)
        train_filepath_1 = os.path.join(data_path, train_filename_1_)
        train_filepath_2 = os.path.join(data_path, train_filename_2_)
        test_filepath = os.path.join(data_path, test_filename)

        train_file1_data, train_file1_labels = read_file(train_filepath_1)
        train_file2_data, train_file2_labels = read_file(train_filepath_2)
        test_file_data, test_file_labels = read_file(test_filepath)

        train_file_data = np.concatenate((train_file1_data, train_file2_data), axis=0)
        train_file_labels = np.concatenate((train_file1_labels, train_file2_labels), axis=0)

        # 数据预处理（白化操作、通道[-1, 1]归一化）
        preprocessed_train = prepare_data(train_file_data)
        preprocessed_test = prepare_data(test_file_data)

        if index == 1:
            train_set = preprocessed_train
            test_set = preprocessed_test
            train_label_set = train_file_labels
            test_label_set = test_file_labels
            index += 1
        else:
            train_set = np.concatenate((train_set, preprocessed_train), axis=0)
            test_set = np.concatenate((test_set, preprocessed_test), axis=0)
            train_label_set = np.concatenate((train_label_set, train_file_labels), axis=0)
            test_label_set = np.concatenate((test_label_set, test_file_labels), axis=0)

    return train_set, test_set, train_label_set, test_label_set

if __name__ == '__main__':
    # data_path = "dataset/BCI Database/Signals/DATA A/"
    # data_set = ['A2', 'A3']
    # # train_set: (80, 27, 1921)    train_label_set: (80, )
    # train_set, test_set, train_label_set, test_label_set = get_EEGSet_BCI_Database(data_path, data_set)
    # print(train_set.shape)
    # print(train_label_set.shape)
    # print(test_set.shape)
    # print(test_label_set.shape)

    # data_path = "dataset/BCI2a/BCICIV_2a_gdf"
    # data_set1 = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07']
    # train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set1)
    # trainset_all = np.concatenate((train_set, test_set))
    # trainlabel_all = np.concatenate((train_label_set, test_label_set))
    # print(trainset_all.shape, trainlabel_all.shape)
    # np.save('data2a/data_bci2a_train_data', trainset_all)
    # np.save('data2a/data_bci2a_train_label', trainlabel_all)

    # data_set2 = ['A08']
    # train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set2)
    # validdata_all = np.concatenate((train_set, test_set))
    # validlabel_all = np.concatenate((train_label_set, test_label_set))
    # print(validdata_all.shape, validlabel_all.shape)
    # np.save('data2a/data_bci2a_valid_data', validdata_all)
    # np.save('data2a/data_bci2a_valid_label', validlabel_all)

    # data_set2 = ['A09']
    # train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set2)
    # testdata_all = np.concatenate((train_set, test_set))
    # testlabel_all = np.concatenate((train_label_set, test_label_set))
    # print(testdata_all.shape, testlabel_all.shape)
    # np.save('data2a/data_bci2a_test_data', testdata_all)
    # np.save('data2a/data_bci2a_test_label', testlabel_all)


    data_path = "dataset/BCI2a/BCICIV_2a_gdf"
    for data_set in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']:
        data_set_get = [data_set]
        train_set, test_set, train_label_set, test_label_set = get_EEGSet(data_path, data_set_get)
        trainset_all = np.concatenate((train_set, test_set))
        trainlabel_all = np.concatenate((train_label_set, test_label_set))

        save_path_data = 'data2a/data_bci2a_' + data_set + '_data.npy'
        save_path_label = 'data2a/data_bci2a_' + data_set + '_label.npy'
        np.save(save_path_data, trainset_all)
        np.save(save_path_label, trainlabel_all)


