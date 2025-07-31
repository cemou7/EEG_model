import numpy as np
import matplotlib.pyplot as plt

# 参数设置
num_channels = 7      # EEG通道数
signal_length = 256   # 每个通道的时间长度
amplitude = 30        # 振幅范围
spacing = 120         # 通道之间的垂直间距

# 设置颜色
colors = ['purple', 'goldenrod', 'steelblue', 'darkmagenta', 'orangered', 'teal', 'olive']

# 生成随机 EEG 信号
np.random.seed(42)
signals = [amplitude * np.random.randn(signal_length) for _ in range(num_channels)]

# 创建偏移后的信号（用于分层显示）
offset_signals = [sig + i * spacing for i, sig in enumerate(signals)]

# 绘图
plt.figure(figsize=(6, 3))
for i, sig in enumerate(offset_signals):
    plt.plot(sig, color=colors[i % len(colors)], linewidth=2)

# 去除坐标轴，设置边框紧凑
plt.axis('off')
plt.tight_layout()
plt.savefig("simulated_eeg.png", dpi=150, bbox_inches='tight')
plt.show()
