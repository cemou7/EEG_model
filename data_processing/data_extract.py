import numpy as np
import mne
from scipy.io import loadmat
import os


class BCICompetition4Set2A:

    def __init__(self, filename, labels_filename=None):
        self.filename = filename
        self.labels_filename = labels_filename

    # return raw
    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info["events"] = events
        cnt.info["artifact_trial_mask"] = artifact_trial_mask
        return cnt

    # 数据提取函数，并将nan数据变为均值数据。
    def extract_data(self):
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))
        # 将原通道重命名为10-20系统中，常用的64个电极位置（通道）
        raw_gdf.rename_channels(
            {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
             'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
             'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
             'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})
        # Load raw data. 加载数据到内存以便接下来的处理
        raw_gdf.load_data()
        # correct nan values. Get data in the given range.      data = ndarray:(22, 672528)
        data = raw_gdf.get_data()

        for i_chan in range(data.shape[0]):     # 遍历 22 channel
            # first set to NaN（缺失值）, then replace nans by nan->mean. 将数组中的所有值设置为nan，然后将这些NaN值替换为该数组的均值。
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            # 获取 NAN 数据下标，将其替换为通道均值 chan_mean
            data[i_chan, mask] = chan_mean

        # 获取事件时间位置，返回的是一个元组
        gdf_events = mne.events_from_annotations(raw_gdf)

        # 利用mne.io.RawArray类重新创建Raw对象，已经没有nan数据了
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
        # remember gdf events
        raw_gdf.info["gdf_events"] = gdf_events
        return raw_gdf

    def extract_events(self, raw_gdf):
        # events_type, events_id
        events, name_to_code = raw_gdf.info["gdf_events"]

        # 确定数据集是训练集T还是验证集E
        if "769" and "770" and "771" and "772" in name_to_code:
            train_set = True
        else:
            train_set = False
            assert (
                # "cue unknown/undefined (used for BCI competition) "
                "783" in name_to_code
            )

        if train_set:
            if self.filename[-8:] == 'A04T.gdf':    # A04T.gdf数据集仅包含眼球运动情况。
                trial_codes = [5, 6, 7, 8]
            else:
                trial_codes = [7, 8, 9, 10]  # the 4 classes, train[7, 8, 9, 10]
        else:
                trial_codes = [7]  # "unknown" class

        # trial_mask = list:603 [false or true] 选择为MI事件的event
        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        # trial_events = {ndarray:(288, 3)} 事件开始时间点，事件id
        trial_events = events[trial_mask]

        assert len(trial_events) == 288, "Got {:d} markers".format(
            len(trial_events)
        )

        if train_set and self.filename[-8:-4] == 'A04T':
            trial_events[:, 2] = trial_events[:, 2] - 4
        else:
            trial_events[:, 2] = trial_events[:, 2] - 6     # 四分类问题，将event_id变为[1, 2, 3, 4]

        # possibly overwrite with markers from labels file. 确保另外给的train_label和train data中的label一样
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes

        # unique_classes:[1 2 3 4]，检查MI类别是否为四分类
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            [1, 2, 3, 4], unique_classes
        ), "Expect 1,2,3,4 as class labels, got {:s}".format(
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
    events = raw_gdb.info['events']
    raw_data = raw_gdb.get_data()
    freqs = raw_gdb.info['sfreq']   # 频率250

    # mi_duration = 1000（4s）, duration_before_mi = -125（-0.5s）, duration_after_mi = 0
    mi_duration = int(freqs * duration)
    duration_before_mi = int(freqs * baseline[0])
    duration_after_mi = int(freqs * baseline[1])

    labels = np.array(events[:, 2])

    trial_data = []
    for i_event in events:  # i_event [time, 0, class]
        # 以第一个MI为例，i_event[0]时间点为92368，提取cue之前0.5s到cue开始后（MI结束）4s，即事件为一个trial的[1.5, 6]
        segmented_data = raw_data[:,
                         int(i_event[0]) + duration_before_mi:int(i_event[0]) + mi_duration + duration_after_mi]
        assert segmented_data.shape[-1] == mi_duration - duration_before_mi + duration_after_mi
        trial_data.append(segmented_data)

    # 对数组进行堆叠. [np.array[22, 1125], np,array[22, 1125]...] -> [288, 22, 1125]
    trial_data = np.stack(trial_data, 0)

    return trial_data, labels


if __name__ == '__main__':
    data_path = "../dataset/BCICIV_2a_gdf"
    data1_path = "../dataset/true_labels_2a"

    train_filename = "A01T.gdf"
    test_filename = "A01E.gdf"
    train_filepath = os.path.join(data_path, train_filename)  # 使用path.join比直接用+显得更规范一些
    test_filepath = os.path.join(data_path, test_filename)
    train_filename = train_filename.replace(".gdf", ".mat")  # 直接用replace函数就可以进行替换
    test_filename = test_filename.replace(".gdf", ".mat")
    train_label_filepath = os.path.join(data1_path, train_filename)
    test_label_filepath = os.path.join(data1_path, test_filename)

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath
    )
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath
    )
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    print(train_cnt.get_data().shape)
    print(test_cnt.get_data().shape)



