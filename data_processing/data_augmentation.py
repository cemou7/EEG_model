import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d      # 线性插值


def trans_example_plot(data_example):
    """
    :param data_example: Tensor of shape [num_channels, sequence_length]
    :return:
    """
    data_example = data_example.cpu().numpy()           # ndarray和tensor数据的实现不同，先转为numpy数据

    Trans = Transform()
    data_warp = Trans.time_warp(data_example)
    data_noise = Trans.gaussian_noise(data_example)
    data_flip = Trans.horizontal_flip(data_example)
    data_permute = Trans.permute_time_segments(data_example)
    data_cutout_resize = Trans.cutout_and_resize(data_example)
    data_crop_resize = Trans.crop_and_resize(data_example)
    data_filter = Trans.average_filter(data_example)

    plt.subplot(8, 1, 1)
    plt.plot(data_example[0, :])
    plt.title("Original signals")

    plt.subplot(8, 1, 2)
    plt.plot(data_warp[0, :])
    plt.title("Time warping")

    plt.subplot(8, 1, 3)
    plt.plot(data_noise[0, :])
    plt.title("Gaussian Noise")

    plt.subplot(8, 1, 4)
    plt.plot(data_flip[0, :])
    plt.title("Horizontal Flip")

    plt.subplot(8, 1, 5)
    plt.plot(data_permute[0, :])
    plt.title("Permute")

    plt.subplot(8, 1, 6)
    plt.plot(data_cutout_resize[0, :])
    plt.title("Cutout & Resize")

    plt.subplot(8, 1, 7)
    plt.plot(data_crop_resize[0, :])
    plt.title("Crop & Resize")

    plt.subplot(8, 1, 8)
    plt.plot(data_filter[0, :])
    plt.title("Average Filter")

    plt.tight_layout()
    plt.show()


class Transform:
    def __init__(self):
        pass

    def time_warp(self, eeg_data):
        """
        Time warping of EEG data: 时空扭曲
        """
        warp_factor = np.random.uniform(0.3, 0.5)
        num_samples = eeg_data.shape[1]
        eeg_warped = np.zeros_like(eeg_data)

        # Generate random warp factor between -warp_factor to +warp_factor
        random_warp = 1 + np.random.uniform(-warp_factor, warp_factor)
        current_length = int(random_warp * num_samples)

        # Generate new time indices based on random warp factor
        old_indices = np.linspace(0, num_samples-1, num_samples)
        new_indices = np.linspace(0, num_samples-1, current_length)

        # Apply warping to each EEG channel
        for i in range(eeg_data.shape[0]):
            if current_length >= eeg_data.shape[1]:
                eeg_warped[i, :] = eeg_data[i, :]
                continue

            # Select data based on current_length
            selected_eeg_data = eeg_data[i, :current_length]

            # 线性插值，实现数据从 1035 -> 1125
            interpolator = interp1d(new_indices, selected_eeg_data, kind='linear', fill_value="extrapolate")
            eeg_warped[i, :] = interpolator(old_indices)

        return eeg_warped

    def gaussian_noise(self, eeg_data, mean=0, std_dev=0.05):
        """
        Adds Gaussian noise to the EEG data: 高斯噪声
        """
        noise = np.random.normal(mean, std_dev, eeg_data.shape)
        return eeg_data + noise

    def horizontal_flip(self, eeg_data):
        """
        flipped horizontally: 同一通道水平翻转
        """
        flipped_eeg = np.flip(eeg_data, axis=1)
        return flipped_eeg

    def permute_time_segments(self, eeg_data):
        """
        Randomly permute segments of EEG data in the time dimension: 打乱同一通道内数据
        """
        segment_length = np.random.randint(3, 10)
        num_segments = eeg_data.shape[1] // segment_length
        permuted_indices = np.random.permutation(num_segments)

        permuted_data = np.concatenate([
            eeg_data[:, i * segment_length:(i + 1) * segment_length] for i in permuted_indices], axis=1)

        # 拼接剩余的数据
        remaining = eeg_data.shape[1] % segment_length
        if remaining:
            permuted_data = np.concatenate([permuted_data, eeg_data[:, -remaining:]], axis=1)

        return permuted_data

    def cutout_and_resize(self, eeg_data):
        """
        Apply the 'Cutout & resize' augmentation to the EEG data: 裁剪
        """
        segment_length = np.random.randint(3, 10)
        segment_length = eeg_data.shape[1] // segment_length

        # 1. Divide the EEG data into segments
        segments = [eeg_data[:, i * segment_length:(i + 1) * segment_length] for i in range(segment_length)]

        # 2. Discard one segment at random: 随机 cutout 一段
        r = np.random.randint(segment_length)
        del segments[r]

        # 3. Concatenate the remaining segments: list -> ndarray
        concatenated_data = np.concatenate(segments, axis=1)

        # 4. Resize the concatenated data to the original length (1125) : 线性插值
        x_old = np.linspace(0, concatenated_data.shape[1] - 1, concatenated_data.shape[1])
        x_new = np.linspace(0, concatenated_data.shape[1] - 1, eeg_data.shape[1])
        resized_data = np.empty_like(eeg_data)

        for i in range(eeg_data.shape[0]):
            interpolator = interp1d(x_old, concatenated_data[i], kind='linear', fill_value='extrapolate')
            resized_data[i] = interpolator(x_new)

        return resized_data

    def crop_and_resize(self, eeg_data):
        """
        :param eeg_data: [channel, time]
        :param size: Shrink size ratio of eeg_data
        :return:
        """
        size = np.random.uniform(0.4, 0.8)
        size = int(eeg_data.shape[1] * size)
        start = np.random.randint(0, eeg_data.shape[1]-size)
        crop_data = eeg_data[:, start:start+size]

        x_old = np.linspace(0, crop_data.shape[1] - 1, crop_data.shape[1])
        x_new = np.linspace(0, crop_data.shape[1] - 1, eeg_data.shape[1])
        resized_data = np.empty_like(eeg_data)

        for i in range(eeg_data.shape[0]):
            interpolator = interp1d(x_old, crop_data[i], kind='linear', fill_value='extrapolate')
            resized_data[i] = interpolator(x_new)

        return resized_data

    def average_filter(self, eeg_data):
        """
        Apply an average filter to the EEG data using convolution: 平均滤波
        """

        k = np.random.randint(3, 11)
        # 构造一个权重总和为1的平均滤波器
        avg_filter = np.ones(k) / k

        filtered_data = np.zeros_like(eeg_data)

        for i in range(eeg_data.shape[0]):
            filtered_data[i] = np.convolve(eeg_data[i], avg_filter, mode='same')

        return filtered_data


