
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 22
        self.increased_dim = 8
        self.final_out_channels = 96
        self.num_classes = 4
        self.num_classes_target = 4
        self.dropout = 0.2
        self.masking_ratio = 0.5
        self.lm = 3 # average length of masking subsequences

        self.kernel_size = 25
        self.stride = 3

        self.TSlength_aligned = 640

        self.CNNoutput_channel = 10

        # EEGNet_ATTEN
        self.afr_reduced_cnn_size = 48
        self.Chans = 22
        self.dropoutRate = 0.5
        self.kernLength1 = 36
        self.kernLength2 = 24
        self.kernLength3 = 18
        self.F1 = 16
        self.D = 2
        self.expansion = 4
        
        self.num_classes = 4
        self.features_len = 960

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4 # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 32

        """For Epilepsy, the target batchsize is 60"""
        self.target_batch_size = 16   # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50
