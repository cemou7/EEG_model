import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne
from typing import List, Tuple
import wget
import sys
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import torch
from mne.io import BaseRaw
from scipy.linalg import sqrtm
from scipy.fftpack import dct, idct
from scipy.signal import firwin, lfilter, filtfilt, butter
from numpy.random import default_rng
import shutil
import re
import matplotlib.pyplot as plt
import pdb

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

class Utils:
    """
    A static class that contains all the functions to generate the dataset and other
    useful functionality
    """
    combinations = {"a": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["FC5", "FC6"]],

                    "b": [["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"]],

                    "c": [["CP1", "CP2"],
                          ["CP3", "CP4"],
                          ["CP5", "CP6"]],

                    "d": [["FC3", "FC4"],
                          ["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP3", "CP4"]],

                    "e": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP1", "CP2"],
                          ["CP3", "CP4"]],

                    "f": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["FC5", "FC6"],
                          ["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP1", "CP2"],
                          ["CP3", "CP4"],
                          ["CP5", "CP6"]]}

    @staticmethod
    def download_data(save_path: str = os.getcwd()) -> str:
        """
        This create a new folder data and download the necessary files
        WARNING: The physionet server is super-slow
        :save_path: data are saved here
        :return: the path
        """
        def bar_progress(current, total, width=80):
            progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        data_url = "https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip"
        data_path = os.path.join(save_path, "eegbci")
        try:
            os.makedirs(data_path)
        except:
            raise Exception("The folder alredy exists")

        wget.download(data_url, os.path.join(data_path, "eegbci.zip"), bar=bar_progress)
        return data_path

    @staticmethod
    def load_data(subjects: List, runs: List, data_path: str) -> List[List[BaseRaw]]:
        """
        Given a list of subjects, a list of runs, and the database path. This function iterates
        over each subject, and subsequently over each run, loads the runs into memory, modifies
        the labels and returns a list of runs for each subject.
        :param subjects: List, list of subjects
        :param runs: List, list of runs
        :param data_path: str, the source path
        :return: List[List[BaseRaw]]
        """
        all_subject_list = []
        subjects = [str(s) for s in subjects]
        runs = [str(r) for r in runs]
        task2 = [4, 8, 12]
        task4 = [6, 10, 14]
        for sub in subjects:
            if len(sub) == 1:
                sub_name = "S"+"00"+sub
            elif len(sub) == 2:
                sub_name = "S"+"0"+sub
            else:
                sub_name = "S"+sub
            sub_folder = os.path.join(data_path, sub_name)
            single_subject_run = []
            for run in runs:
                if len(run) == 1:
                    path_run = os.path.join(sub_folder, sub_name+"R"+"0"+run+".edf")
                else:
                    path_run = os.path.join(sub_folder, sub_name+"R"+ run +".edf")
                raw_run = read_raw_edf(path_run, preload=True)
                len_run = np.sum(raw_run._annotations.duration)
                if len_run > 124:
                    print(sub)
                    raw_run.crop(tmax=124)

                """
                B indicates baseline
                L indicates motor imagination of opening and closing left fist;
                R indicates motor imagination of opening and closing right fist;
                LR indicates motor imagination of opening and closing both fists;
                F indicates motor imagination of opening and closing both feet.
                """

                if int(run) in task2:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "L"
                        if an == "T2":
                            raw_run.annotations.description[index] = "R"
                if int(run) in task4:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "LR"
                        if an == "T2":
                            raw_run.annotations.description[index] = "F"
                single_subject_run.append(raw_run)
            all_subject_list.append(single_subject_run)
        return all_subject_list

    @staticmethod
    def concatenate_runs(list_runs: List[List[BaseRaw]]) -> List[BaseRaw]:
        """
        Concatenate a list of runs
        :param list_runs: List[List[BaseRaw]],  list of raw
        :return: List[BaseRaw], list of concatenate raw
        """
        raw_conc_list = []
        for subj in list_runs:
            raw_conc = concatenate_raws(subj)
            raw_conc_list.append(raw_conc)
        return raw_conc_list

    @staticmethod
    def del_annotations(list_of_subraw: List[BaseRaw]) -> List[BaseRaw]:
        """
        Delete "BAD boundary" and "EDGE boundary" from raws
        :param list_of_subraw: list of raw
        :return: list of raw
        """
        list_raw = []
        for subj in list_of_subraw:
            indexes = []
            for index, value in enumerate(subj.annotations.description):
                if value == "BAD boundary" or value == "EDGE boundary":
                    indexes.append(index)
            subj.annotations.delete(indexes)
            list_raw.append(subj)
        return list_raw

    @staticmethod
    def eeg_settings(raws:  List[BaseRaw]) -> List[BaseRaw]:
        """
        Standardize montage of the raws
        :param raws: List[BaseRaw] list of raws
        :return: List[BaseRaw] list of standardize raws
        """
        raw_setted = []
        for subj in raws:
            eegbci.standardize(subj)
            montage = make_standard_montage('standard_1005')
            subj.set_montage(montage)
            raw_setted.append(subj)

        return raw_setted

    @staticmethod
    def filtering(list_of_raws: List[BaseRaw]) -> List[BaseRaw]:
        """
        Perform a band_pass and a notch filtering on raws, UNUSED!
        :param list_of_raws:  list of raws
        :return: list of filtered raws
        """
        raw_filtered = []
        for subj in list_of_raws:
            if subj.info["sfreq"] == 160.0:
                subj.filter(4.0, 38.0, fir_design='firwin', skip_by_annotation='edge')
                subj.notch_filter(freqs=60)
                raw_filtered.append(subj)
            else:
                subj.filter(1.0, (subj.info["sfreq"] / 2) - 1, fir_design='firwin',
                            skip_by_annotation='edge')
                subj.notch_filter(freqs=60)
                raw_filtered.append(subj)

        return raw_filtered

    @staticmethod
    def select_channels(raws: List[BaseRaw], ch_list: List, allselect = False ) -> List[BaseRaw]:
        """
        Slice channels
        :raw: List[BaseRaw], List of Raw EEG data
        :ch_list: List
        :return: List[BaseRaw]
        """
        if allselect == True:
            return raws
        else:
            s_list = []
            for raw in raws:
                s_list.append(raw.pick_channels(ch_list))

            return s_list

    @staticmethod
    def epoch(raws: List[BaseRaw], exclude_base: bool =False, num_class: int = 5,
              tmin: int =0, tmax: int =4) -> Tuple[np.ndarray, List[str]]:
        """
        Split the original BaseRaw into numpy epochs
        :param raws: List[BaseRaw]
        :param exclude_base: bool, If True exclude baseline
        :param tmin: int, Onset
        :param tmax: int, Offset
        :return: np.ndarray (Raw eeg datas in numpy format) List (a List of strings)
        """
        xs = list()
        ys = list()
        for raw in raws:
            if num_class == 5:
                if exclude_base:
                    event_id = dict(F=0, L=1, LR=2, R=3)
                else:
                    event_id = dict(B=0, F=1, L=2, LR=3, R=4)
            elif num_class == 3:
                if exclude_base:
                    event_id = dict(L=0, R=1)
                else:
                    event_id = dict(B=0, L=1, R=2)
            else:
                if exclude_base:
                    event_id = dict(L=0, R=1, F=2)
                else:
                    event_id = dict(B=0, L=1, R=2, F=3)               
            tmin, tmax = tmin, tmax
            events, _ = mne.events_from_annotations(raw, event_id=event_id)

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                   exclude='bads')
            epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)

            y = list()
            for index, data in enumerate(epochs):
                y.append(epochs[index]._name)

            xs.append(np.array([epoch for epoch in epochs]))
            ys.append(y)

        return np.concatenate(tuple(xs), axis=0), [item for sublist in ys for item in sublist]

    @staticmethod
    def cut_width(data):
        new_data = np.zeros((data.shape[0], data.shape[1], data.shape[2] - 1))
        for index, line in enumerate(data):
            new_data[index] = line[:, : -1]

        return new_data

    @staticmethod
    def load_sub_by_sub(subjects, data_path, name_single_sub):
        xs = list()
        ys = list()
        for sub in subjects:
            xs.append(Utils.cut_width(np.load(os.path.join(data_path, "x" + name_single_sub + str(sub) + ".npy"))))
            ys.append(np.load(os.path.join(data_path, "y" + name_single_sub + str(sub) + ".npy")))
        return xs, ys

    @staticmethod
    def scale_sub_by_sub(xs, ys):
        for sub_x, sub_y, sub_index in zip(xs, ys, range(len(xs))):
            for sample_index, x_data in zip(range(sub_x.shape[0]), sub_x):
                xs[sub_index][sample_index] = minmax_scale(x_data, axis=1)

        return xs, ys


    @staticmethod
    def to_one_hot(labels, classes):
        label_indices = np.array([np.where(classes == label)[0][0] for label in labels])
        one_hot_labels = np.eye(len(classes))[label_indices]

        # 转换为单个类别标签
        train_label_single = np.argmax(one_hot_labels, axis=1)
        return train_label_single

    @staticmethod
    def train_test_split(x, y, perc):
        rng = default_rng()
        test_x = list()
        train_x = list()
        train_y = list()
        test_y = list()
        for sub_x, sub_y in zip(x, y):
            how_many = int(len(sub_x) * perc)
            indexes = np.arange(0, len(sub_x))
            choices = rng.choice(indexes, how_many, replace=False)
            for sample_x, sample_y, index in zip(sub_x, sub_y, range(len(sub_x))):
                if index in choices:
                    test_x.append(sub_x[index])
                    test_y.append(sub_y[index])
                else:
                    train_x.append(sub_x[index])
                    train_y.append(sub_y[index])
        return np.dstack(tuple(train_x)), np.dstack(tuple(test_x)), np.array(train_y), np.array(test_y)
    
    @staticmethod
    def preprocess_ea(data):
        R_bar = np.zeros((data.shape[1], data.shape[1]))
        for i in range(len(data)):
            R_bar += np.dot(data[i], data[i].T)
        R_bar_mean = R_bar / len(data)
        # assert (R_bar_mean >= 0 ).all(), 'Before squr,all element must >=0'

        for i in range(len(data)):
            data[i] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), data[i])
        return data
    
    @staticmethod
    def data_norm(data):
        """
        对数据进行归一化
        :param data:   ndarray ,shape[N,channel,samples]
        :return:
        """
        data_copy = np.copy(data)
        for i in range(len(data)):
            data_copy[i] = data_copy[i] / np.max(abs(data[i]))

        return data_copy

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
        # nyq_freq = 0.5 * fs
        # low = low_cut_hz / nyq_freq
        # high = high_cut_hz / nyq_freq

        # win = firwin(filt_order, [low, high], window='blackman', ass_zero='bandpass')
        win = firwin(filt_order, [low_cut_hz, high_cut_hz], window='blackman', fs=fs, pass_zero='bandpass')

        data_bandpassed = lfilter(win, 1, data)
        if zero_phase:
            data_bandpassed = filtfilt(win, 1, data)
        return data_bandpassed

    def bci_prepare_data(data):
        # [-1,1]

        data_preprocss = Utils.data_norm(data)
        data_ea = Utils.preprocess_ea(data_preprocss)

        data_pre = np.expand_dims(data_ea, axis=1)

        return data_pre

    def prepare_data(data):
        # [-1,1]

        data_preprocss = Utils.data_norm(data)
        data_ea = Utils.preprocess_ea(data_preprocss)

        return data_ea

    def preprocess4mi(data):  # data: ndarray with shape, nums, chans, samples
        # data_filter = bandpass_cnt(data,  low_cut_hz=4, high_cut_hz=38, fs=250)
        data_preprocessed = Utils.bci_prepare_data(data)
        return data_preprocessed

    @staticmethod
    def extract_segment_trial(raw_gdb, baseline=(0, 0), duration=4):
        '''
        get segmented data and corresponding labels from raw_gdb.
        :param raw_gdb: raw data
        :param baseline: unit: second. baseline for the segment data. The first value is time before cue.
                        The second value is the time after the Mi duration. Positive values represent the time delays,
                        negative values represent the time lead.
        :param duration: unit: seconds. mi duration time
        :return: array data: trial data, labels
        '''
        events = raw_gdb.info['temp']['events']
        raw_data = raw_gdb.get_data()
        freqs = raw_gdb.info['sfreq']
        mi_duration = int(freqs * duration)
        duration_before_mi = int(freqs * baseline[0])
        duration_after_mi = int(freqs * baseline[1])

        labels = np.array(events[:, 2])

        trial_data = []
        for i_event in events:  # i_event [time, 0, class]
            segmented_data = raw_data[:,
                            int(i_event[0]) + duration_before_mi:int(i_event[0]) + mi_duration + duration_after_mi]
            assert segmented_data.shape[-1] == mi_duration - duration_before_mi +duration_after_mi
            trial_data.append(segmented_data)
        trial_data = np.stack(trial_data, 0)

        return trial_data, labels
    
    def shuffle_data(data, labels):
        # 打乱顺序
        perm = np.random.permutation(data.shape[0])

        # 按照打乱顺序重新排列数据和标签
        shuffled_data = data[perm]
        shuffled_labels = labels[perm]
        return data, labels

    def save_train_valid_acc_loss_fig(path):
        """
        path:logging source file
        return: train_valid_acc_loss_fig

        """

        with open(path, 'r') as file:
            log_content = file.read()
        train_acc = re.findall(r'train_acc\s+(\d+\.\d+)', log_content)
        train_acc = [float(match) for match in train_acc]
        train_loss = re.findall(r'train_loss\s+(\d+\.\d+)', log_content)
        train_loss = [float(match) for match in train_loss]
        valid_acc = re.findall(r'valid_acc\s+(\d+\.\d+)', log_content)
        valid_acc = [float(match) for match in valid_acc]
        valid_loss = re.findall(r'valid_loss\s+(\d+\.\d+)', log_content)
        valid_loss = [float(match) for match in valid_loss]

        matplotlib.use("Agg")
        plt.style.use('seaborn')

        SMALL_SIZE = 20
        MEDIUM_SIZE = 35
        BIGGER_SIZE = 45

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.subplot(1,2,1, title="train and valid Accuracy")
        plt.plot(range(len(train_acc)), train_acc, label="Train", linewidth=4)
        plt.plot(range(len(valid_acc)), valid_acc, label="Valid", linewidth=4)
        plt.legend(loc='lower right')
        
        plt.subplot(1,2,2, title="train and valid Loss")
        plt.plot(range(len(train_loss)), train_loss, label="Train", linewidth=4)
        plt.plot(range(len(valid_loss)), valid_loss, label="Valid", linewidth=4)
        plt.legend(loc='upper right')
        
        # plt.subplots_adjust(wspace=0.5)
        plt.savefig('./train_valid_acc_loss.png', dpi = 300)
        plt.close()

    def save_acc_fig(path):
        """
        path:logging source file
        return: train_valid_acc_fig

        """
        with open(path, 'r') as file:
            log_content = file.read()
        train_acc = re.findall(r'train_acc\s+(\d+\.\d+)', log_content)
        train_acc = [float(match) for match in train_acc]
        valid_acc = re.findall(r'valid_acc\s+(\d+\.\d+)', log_content)
        valid_acc = [float(match) for match in valid_acc]
        plt.figure()
        plt.plot(range(len(train_acc)), train_acc, label='train_acc', linewidth=4)
        plt.plot(range(len(valid_acc)), valid_acc, label='valid_acc', linewidth=4)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        plt.savefig('./acc.png')
        plt.close()

    def save_loss_fig(path):
        """
        path:logging source file
        return: train_valid_loss_fig

        """
        with open(path, 'r') as file:
            log_content = file.read()
        train_loss = re.findall(r'train_loss\s+(\d+\.\d+)', log_content)
        train_loss = [float(match) for match in train_loss]
        valid_loss = re.findall(r'valid_loss\s+(\d+\.\d+)', log_content)
        valid_loss = [float(match) for match in valid_loss]
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label='train_loss', linewidth=4)
        plt.plot(range(len(valid_loss)), valid_loss, label='valid_loss', linewidth=4)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        plt.savefig('./loss.png', dpi = 300)
        plt.close()

    def tSNE_plot(data, label, tSNE_path):
        data = data.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(data)
        
        x_min, x_max = np.min(result, 0), np.max(result, 0)
        result = (result - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(result.shape[0]):
            plt.text(result[i, 0], result[i, 1], str(label[i]),
                    color=plt.cm.Set1(label[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE embedding of the digits [epoch {}]'.format(tSNE_path[68:72]))
        plt.savefig(tSNE_path)
        plt.close()

    def plot_embedding(data, label, title):
        data = data.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
    
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                    color=plt.cm.Set1(label[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.savefig('./{:s}.png'.format(title), dpi=300)
        plt.close()

    def label_smooth(ground_truth_angle):
        smoothed_label = []
        for ground_truth in ground_truth_angle:
            label = np.zeros(9)
            curve = 2
            for i in range(label.shape[0]):
                if abs(i - ground_truth) <= curve:
                    label[i] = max(label[i], np.exp(-1 * (i - ground_truth) ** 2 / curve ** 2))
                elif abs(i + 9 - ground_truth) <= curve:
                    label[i] = max(label[i], np.exp(-1 * (i + 9 - ground_truth) ** 2 / curve ** 2))
                elif abs(i - 9 - ground_truth) <= curve:
                    label[i] = max(label[i], np.exp(-1 * (i - 9 - ground_truth) ** 2 / curve ** 2))

            label = (label / np.sum(label)).astype(np.float32)
            smoothed_label.append(label)
        return np.array(smoothed_label)

    def reverse_label_smooth(smoothed_label):
        ground_truth_angle = []
        for label in smoothed_label:
            ground_truth = torch.argmax(label.clone().detach())
            ground_truth_angle.append(ground_truth.item())
        return torch.tensor(ground_truth_angle).to(smoothed_label.device)

if __name__ == "__main__":
    x_MI, y_MI = Utils.loadData_MI()
    pdb.set_trace()


