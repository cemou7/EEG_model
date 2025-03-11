import os
import sys
from pathlib import Path

sys.path.append("/home/work3/wkh/CL-Model")
from data_processing.general_processor import Utils
import numpy as np
import mne
from scipy.io import loadmat
import shutil
import random
import pdb


class PHYSIONET:

    def __init__(self, filename, labels_filename=None):
        self.filename = filename
        self.labels_filename = labels_filename

    def produceData_MI(self):
        """
        load motor imagery dataset 
        """
        channels = ["C3", "C4", "P1", "P2", "PO7", "PO8", "O1", "O2"]
        # channels = ["C3", "C4", "P3", "P4", "PO7", "PO8", "O1", "O2"]
        data_path = "/home/work/kober/EEG-GPT/dataset/original/physionet"
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 110) if n not in exclude]
        # Utils.prodeceData_MI(channels, data_path, subjects)
        runs = [4, 8, 12]  #left hand, reght hand, baseline
        save_path = "/home/work/kober/EEG-GPT/dataset/processed_data/physionet_8ch"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        for sub in subjects:
            x, y = Utils.epoch(Utils.select_channels
                (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
                Utils.load_data(subjects=[sub], runs=runs, data_path=data_path)))), channels),
                exclude_base=False)

            np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
            np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
            
    def loadData_MI(subjects):
        channels_path = "/home/work/kober/EEG-GPT/dataset/processed_data/physionet_8ch"
        data_x = list()
        data_y = list()
        sub_name = "_sub_"
        xs, ys = Utils.load_sub_by_sub(subjects, channels_path, sub_name)
        data_x.append(np.concatenate(xs))
        data_y.append(np.concatenate(ys))
        
        data = np.concatenate(data_x)
        label = np.concatenate(data_y)
        classes = np.array(['B', 'L', 'R'])
        to_one_label = Utils.to_one_hot(label, classes)
        return data, to_one_label
    
    def train_test_data(train_rate, test_rate):
        assert train_rate + test_rate == 1,"Train rate and test rate should sum up to 1"
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 110) if n not in exclude]
        random.shuffle(subjects)
        channels = Utils.combinations["obj1"]
        
        train_subjects = subjects[:int(len(subjects)*train_rate)]
        test_subjects = subjects[int(len(subjects)*train_rate):]

        selected_class = ["L", "R", "LR", "F"]
        source_path = "dataset/PHYSIONET/paper/"
        train_data, train_labels = Utils.get_data_list(channels, train_subjects, source_path, selected_class)
        test_data, test_labels = Utils.get_data_list(channels, test_subjects, source_path, selected_class)
        
        return train_data, test_data, train_labels, test_labels

    def train_val_test_data(train_rate, valid_rate, test_rate):
        assert train_rate + valid_rate + test_rate == 1,"Train rate , valid rate and test rate should sum up to 1"
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 110) if n not in exclude]
        random.shuffle(subjects)
        channels = Utils.combinations["obj1"]
        
        train_subjects = subjects[:int(len(subjects)*train_rate)]
        valid_subjects = subjects[int(len(subjects)*train_rate):int(len(subjects)*(train_rate + valid_rate))]
        test_subjects = subjects[int(len(subjects)*(train_rate + valid_rate)):]
        
        train_data, train_labels = PHYSIONET.loadData_MI(train_subjects)
        valid_data, valid_labels = PHYSIONET.loadData_MI(valid_subjects)
        test_data, test_labels = PHYSIONET.loadData_MI(test_subjects)
        
        return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

class BCICompetition4Set2A:

    def __init__(self, labels_filename=None):
        self.filename = None
        self.data = None
        self.labels = None
        self.labels_filename = labels_filename

    def load(self, subjects):
        # 获取 preprocess_2a.py 所在的目录
        SCRIPT_DIR = Path(__file__).resolve().parent

        # 计算原始数据的 **绝对路径**
        data_path = SCRIPT_DIR / "dataset/BCI2a/BCICIV_2a_gdf"
        # data_path = "/home/work3/wkh/CL-Model/dataset/BCI2a/BCICIV_2a_gdf/"
        index = 1
        for subject_id in subjects:
            train_filename = "A{:02d}T.gdf".format(subject_id)
            self.filename = os.path.join(data_path, train_filename)

            train_cnt = self.extract_data()
            events, artifact_trial_mask = self.extract_events(train_cnt)
            train_cnt.info["temp"] = dict()
            train_cnt.info["temp"]["events"] = events
            train_cnt.info["temp"]["artifact_trial_mask"] = artifact_trial_mask

            train_x, labels_x = Utils.extract_segment_trial(train_cnt)
            if index == 1:
                self.data = train_x
                self.labels = labels_x
            else:
                self.data = np.concatenate((self.data, train_x), axis=0)
                self.labels = np.concatenate((self.labels, labels_x), axis=0)
            index += 1
            
        return self.data, self.labels

    def extract_data(self):
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))
        raw_gdf.rename_channels(
            {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
             'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
             'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
             'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})
        raw_gdf.load_data()
        # correct nan values
        data = raw_gdf.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean
        gdf_events = mne.events_from_annotations(raw_gdf)
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
        # remember gdf events
        # raw_gdf.info["gdf_events"] = gdf_events
        raw_gdf.info["temp"] = gdf_events
        return raw_gdf

    def extract_events(self, raw_gdf):
        # all events
        events, name_to_code = raw_gdf.info["temp"]
        if "769" and "770" and "771" and "772" in name_to_code :
            train_set = True
        else:
            train_set = False
            assert (
                # "cue unknown/undefined (used for BCI competition) "
                "783" in name_to_code
            )

        if train_set:
            if self.filename[-8:] == 'A04T.gdf':
                trial_codes = [5, 6, 7, 8]
            else:
                trial_codes = [7, 8, 9, 10]  # the 4 classes, train[7, 8, 9, 10]
        else:
                trial_codes = [7]  # "unknown" class
        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]

        assert len(trial_events) == 288, "Got {:d} markers".format(
            len(trial_events)
        )
        # print('self.filename[-8:-5]: ', self.filename[-8:-5])
        if train_set and self.filename[-8:-4] == 'A04T':
            trial_events[:, 2] = trial_events[:, 2] - 5
        else:
            trial_events[:, 2] = trial_events[:, 2] - 7

        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:  # 确保另外给的train_label和train data中的label一样
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            [0, 1, 2, 3], unique_classes
        ), "Expect 0, 1, 2, 3 as class labels, got {:s}".format(
            str(unique_classes)
        )

        # now also create 0-1 vector for rejected trials
        if train_set and self.filename[-8:-5] == 'A04':
            trial_start_events = events[events[:, 2] == 4]  # 768 start a trail
        else:
            trial_start_events = events[events[:, 2] == 6]  # 768 start a trail
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]

        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1
        return trial_events, artifact_trial_mask
    
    def train_val_test_data(train_rate, valid_rate, test_rate):
            
            assert train_rate + valid_rate + test_rate == 1,"Train rate , valid rate and test rate should sum up to 1"
            subjects = [n for n in np.arange(1, 10)]
            random.shuffle(subjects)
            
            train_subjects = subjects[:int(len(subjects)*train_rate)]
            valid_subjects = subjects[int(len(subjects)*train_rate):int(len(subjects)*(train_rate + valid_rate))]
            test_subjects = subjects[int(len(subjects)*(train_rate + valid_rate)):]
            
            train_data, train_labels = BCICompetition4Set2A().load(train_subjects)
            valid_data, valid_labels = BCICompetition4Set2A().load(valid_subjects)
            test_data, test_labels = BCICompetition4Set2A().load(test_subjects)
            
            return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

class BCICompetition4Set2B:

    def __init__(self, labels_filename=None):
        self.filename = None
        self.data = None
        self.labels = None
        self.labels_filename = labels_filename

    def load(self, subjects):
        data_path = "/home/work3/wkh/CL-Model/dataset/BCI2b/BCICIV_2b_gdf/"
        index = 1
        for subject_id in subjects:
            for sub_id in range(1, 4):
                train_filename = "B{:02d}{:02d}T.gdf".format(subject_id, sub_id)
                print(train_filename)
                self.filename = os.path.join(data_path, train_filename)
                
                train_cnt = self.extract_data()
                events, artifact_trial_mask = self.extract_events(train_cnt)
                train_cnt.info["temp"] = dict()
                train_cnt.info["temp"]["events"] = events
                train_cnt.info["temp"]["artifact_trial_mask"] = artifact_trial_mask
                
                train_x, labels_x = Utils.extract_segment_trial(train_cnt)
                
                if index == 1:
                    self.data = train_x
                    self.labels = labels_x
                else:
                    self.data = np.concatenate((self.data, train_x), axis=0)
                    self.labels = np.concatenate((self.labels, labels_x), axis=0)
                index += 1
                
        return self.data, self.labels

    def extract_data(self):
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))
        channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        raw_pick = raw_gdf.pick('eeg', exclude=channels_to_remove)
        raw_pick.rename_channels({'EEG:C3': 'C3', 'EEG:Cz': 'Cz', 'EEG:C4': 'C4'})
        raw_pick.load_data()
        # correct nan values
        data = raw_pick.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean
        gdf_events = mne.events_from_annotations(raw_pick)
        raw_pick = mne.io.RawArray(data, raw_pick.info, verbose="ERROR")
        # remember gdf events
        # raw_gdf.info["gdf_events"] = gdf_events
        raw_pick.info["temp"] = gdf_events
        return raw_pick

    def extract_events(self, raw_gdf):
        # all events
        events, name_to_code = raw_gdf.info["temp"]
        if "769" and "770" in name_to_code :
            train_set = True
        else:
            train_set = False
            assert (
                # "cue unknown/undefined (used for BCI competition) "
                "783" in name_to_code
            )

        if train_set:
            if self.filename[-10:] == 'B0102T.gdf':
                trial_codes = [4, 5]
            else:
                trial_codes = [10, 11]
        else:
                trial_codes = [7]  # "unknown" class
        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]
        assert len(trial_events) == 120 or 160, "Got {:d} markers".format(
            len(trial_events)
        )
        # print('self.filename[-8:-5]: ', self.filename[-8:-5])
        if self.filename[-10:] == 'B0102T.gdf':
            trial_events[:, 2] = trial_events[:, 2] - 4
        else:
            trial_events[:, 2] = trial_events[:, 2] - 10
        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:  # 确保另外给的train_label和train data中的label一样
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            [0, 1], unique_classes
        ), "Expect 0,1 as class labels, got {:s}".format(
            str(unique_classes)
        )

        # now also create 0-1 vector for rejected trials
        if train_set and self.filename[-10:] == 'B0102T.gdf':
            trial_start_events = events[events[:, 2] == 3]  # 768 start a trail
        else:
            trial_start_events = events[events[:, 2] == 9]  # 768 start a trail
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]

        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1
        return trial_events, artifact_trial_mask
    
    def train_val_test_data(train_rate, valid_rate, test_rate):
            
            assert train_rate + valid_rate + test_rate == 1,"Train rate , valid rate and test rate should sum up to 1"
            subjects = [n for n in np.arange(1, 10)]
            random.shuffle(subjects)
            
            train_subjects = subjects[:int(len(subjects)*train_rate)]
            valid_subjects = subjects[int(len(subjects)*train_rate):int(len(subjects)*(train_rate + valid_rate))]
            test_subjects = subjects[int(len(subjects)*(train_rate + valid_rate)):]
            
            train_data, train_labels = BCICompetition4Set2B().load(train_subjects)
            valid_data, valid_labels = BCICompetition4Set2B().load(valid_subjects)
            test_data, test_labels = BCICompetition4Set2B().load(test_subjects)
            
            return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


if __name__ == '__main__':
    # train_data_physio, train_labels_physio, test_data_physio, test_labels_physio = PHYSIONET.train_test_data(0.8, 0.2)
    train_data_bci2a, train_labels_bci2a, valid_data_bci2a, valid_labels_bci2a, test_data_bci2a, test_labels_bci2a = BCICompetition4Set2A.train_val_test_data(0.8, 0.2, 0)
    # train_data_bci2b, train_labels_bci2b, valid_data_bci2b, valid_labels_bci2b, test_data_bci2b, test_labels_bci2b = BCICompetition4Set2B.train_val_test_data(0.8, 0.2, 0)
    # print(train_data_physio.shape)
    # print(train_data_bci2b.shape)
    print(train_data_bci2a.shape)


