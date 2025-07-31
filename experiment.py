from logging import Logger
import os
import random
import shutil
# from umap import UMAP
import sys
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
from openTSNE import TSNE
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from data_processing.data_augmentation import Transform, trans_example_plot
from model_set.EEGNet_ATTEN import EEGNet_ATTEN
from model_set.EEGNet_ATTEN_Multi_Task import EEGNet_ATTEN_Multi_Task
from model_set.LMAD import LMDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from model_set.LMAD_Multi_Task import LMDA_Multi_Task
# from model_set.LMDA_net_GCN import LMDA
# from model_set.lmda_band_graph_conv import LMDA_GCN
# from model_set.net_lmda import LMDA

sys.path.append("/home/work3/wkh/CL-Model")

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

# 模型初始化的部分
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ========== 1. 加载模型结构 ==========
def load_model(model_name, model_path, num_classes,device):
    if model_name == 'LMDANet':
        model = LMDA(num_classes=num_classes, chans=22, samples=1000,
                     channel_depth1=24, channel_depth2=9, kernel=75,
                     depth=9, ave_depth=1, avepool=25
                    #  , classification=True
                     )        
    elif model_name == 'LMDA_Multi_Task':
        model = LMDA_Multi_Task(num_classes1=2,num_classes2=4, chans=22, samples=1000,
                       channel_depth1=24,
                       channel_depth2=9,
                       kernel=75, depth=9,
                       ave_depth=1, avepool=25
                       )
    elif model_name == 'EEGNet_ATTEN':
        model = EEGNet_ATTEN(Chans=22,kernLength1=36,kernLength2=24,kernLength3=18,F1=16,D=2,classification_num=4,DOR=0.5)
    elif model_name == 'EEGNet_ATTEN_Multi_Task':
        model = EEGNet_ATTEN_Multi_Task(Chans=22,kernLength1=36,kernLength2=24,kernLength3=18,
                                          F1=16,D=2,num_classes1=2,num_classes2=4,DOR=0.5)
        # model = LMDA(num_classes=4, chans=22, samples=1000,
        #              channel_depth1=24, channel_depth2=9, kernel=75,
        #              depth=9, ave_depth=1, avepool=25)
        # model = LMDA(num_classes=4, chans=22, samples=1125,
        #         channel_depth1=24, channel_depth2=9, kernel=75,
        #         depth=9, ave_depth=1, avepool=25)
    elif model_name == 'EEGNet':
        from model_set.net import EEGNet
        model = EEGNet(num_classes=4, chans=22, samples=1000, kernLength=512//2)
    elif model_name == 'EEG_Conformer':
        from model_set.conformer import Conformer
        model = Conformer(channel=22, n_classes=4, cla=True)
    elif model_name == 'dfformer':
        from model_set.dfformer import get_dfformer_model
        model = get_dfformer_model()
    elif model_name == 'PatchTST':
        from PatchTST_supervised.models import PatchTST
        args = PatchTST.set_parser()
        model = PatchTST.PatchTST(args, cla=True)    
    # elif model_name == 'LMDANet_GCN':
    #     model = LMDA_GCN(num_classes=4, chans=22, samples=1000,
    #                  channel_depth1=24, channel_depth2=9, kernel=75,
    #                  depth=9, ave_depth=1, avepool=25, classification=True,adj_init=adj_init) 
    else:
        raise ValueError("Unsupported model name!")

    # 加载权重参数
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 放入设备 & 切换eval模式
    model = model.to(device)
    model.eval()
    return model




def predict_and_evaluate(model, test_dataloader, device, model_name='LMDANet'):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device).float()
            y = y.to(device).long()

            if model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDA_Multi_Task']:
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
def predict_and_evaluate_multi_task(model, test_dataloader, device, task=1):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device).float()
            y = y.to(device).long()

            # 适配不同模型输入形状  LMDA_Multi_Task需要扩充维度
            # X = X.unsqueeze(1)  # 一般你的模型都需要 (B, 1, C, T)

            # 前向传播
            features = model.backbone(X)

            if task == 1:
                logits = model.classifier1(features)
            elif task == 2:
                logits = model.classifier2(features)
            else:
                raise ValueError("task 必须是 1（二分类）或 2（四分类）")

            # 预测类别
            predicted = torch.argmax(logits, dim=1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # 准确率计算
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    acc = correct / len(all_labels)

    print(f"\n[Task{task}] 测试集准确率: {acc:.4f} ({correct}/{len(all_labels)} correct)")
    print(f"\n[Task{task}] 实际标签分布: {dict(Counter(all_labels))}")
    print(f"\n[Task{task}] 预测结果分布: {dict(Counter(all_preds))}")

    return all_preds, all_labels, acc

# data augmentation
def transform(eeg_data, mode):
    eeg_data = eeg_data.cpu().numpy()

    Trans = Transform()
    if mode == 'time_warp':
        eeg_data = Trans.time_warp(eeg_data)
    elif mode == 'gaussian_noise':
        eeg_data = Trans.gaussian_noise(eeg_data)
    elif mode == 'horizontal_flip':
        eeg_data = Trans.horizontal_flip(eeg_data)
    elif mode == 'permute':
        eeg_data = Trans.permute_time_segments(eeg_data)
    elif mode == 'cutout_resize':
        eeg_data = Trans.cutout_and_resize(eeg_data)
    elif mode == 'crop_resize':
        eeg_data = Trans.crop_and_resize(eeg_data)
    elif mode == 'move_avg':
        eeg_data = Trans.average_filter(eeg_data)
    else:
        print("Error")

    return eeg_data


# draw different transform
def transform_plot(train_dataloader):
    data_example, label_example = next(iter(train_dataloader))
    print(data_example.shape)
    trans_example_plot(data_example[0])


# SimCLR paper contrast_loss code
def contrast_loss(x, temperature):
    LARGE_NUM = 1e9
    x = F.normalize(x, dim=-1)

    num = int(x.shape[0] / 2)
    hidden1, hidden2 = torch.split(x, num)

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0, num).to('cuda')
    masks = F.one_hot(torch.arange(0, num), num).to('cuda')

    logits_aa = torch.matmul(hidden1, hidden1_large.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / temperature

    criterion = torch.nn.CrossEntropyLoss()
    loss_a = criterion(torch.cat([logits_ab, logits_aa], 1), labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], 1), labels)

    loss = torch.mean(loss_a + loss_b)
    return loss


# 1-cosine similarity loss  
def cosine_distance_loss(eeg_signal1, eeg_signal2):
    """
    Calculate the 1-cosine similarity loss between two EEG signals.
    
    Args:
    eeg_signal1, eeg_signal2 (torch.Tensor): tensors of shape [bs, num_patch, n_vars, patch_len]

    Returns:
    torch.Tensor: The mean cosine distance loss across all samples.
    """
    # Flatten the tensors except for the last dimension
    eeg_signal1_flat = eeg_signal1.reshape(eeg_signal1.shape[0], -1)
    eeg_signal2_flat = eeg_signal2.reshape(eeg_signal2.shape[0], -1)

    # Calculate cosine similarity
    cosine_sim = F.cosine_similarity(eeg_signal1_flat, eeg_signal2_flat, dim=1)

    # Calculate mean cosine distance
    cosine_distance = 1 - cosine_sim
    return cosine_distance.mean()


# def ssl_exponential_contrastive_loss(feat_online_pos, feat_target_pos, feat_target_neg, delta=0.3, sigma=2.0):
#     # L2 normalize
#     def l2_normalize(x):
#         return x / (x.norm(dim=1, keepdim=True) + 1e-8)

#     feat_online_pos = l2_normalize(feat_online_pos)
#     feat_target_pos = l2_normalize(feat_target_pos)
#     feat_target_neg = l2_normalize(feat_target_neg)

#     # 平均每个样本的平方距离，然后整体求均值
#     pos_dist = ((feat_online_pos - feat_target_pos) ** 2).sum(dim=1).mean()
#     neg_dist = ((feat_online_pos - feat_target_neg) ** 2).sum(dim=1).mean()

#     exp_pos = torch.exp(-pos_dist / (2 * sigma ** 2))
#     exp_neg = torch.exp(-neg_dist / (2 * sigma ** 2))

#     loss = delta * exp_pos - exp_neg
#     return loss

# def ssl_margin_exponential_contrastive_loss(
#     feat_anchor,        # anchor 特征（batch_size, dim）
#     feat_positive,      # 正样本特征（batch_size, dim）
#     feat_negative,      # 负样本特征（batch_size, dim）
#     sigma=2.0,
#     margin=0.1
# ):
#     """
#     Exponential contrastive loss with margin:
#     loss = max(0, exp(-neg_dist / 2σ²) - exp(-pos_dist / 2σ²) + margin)
#     """
#     # L2 normalize
#     feat_anchor = F.normalize(feat_anchor, p=2, dim=1)
#     feat_positive = F.normalize(feat_positive, p=2, dim=1)
#     feat_negative = F.normalize(feat_negative, p=2, dim=1)

#     # 欧氏平方距离
#     pos_dist = ((feat_anchor - feat_positive) ** 2).sum(dim=1)  # shape: (batch,)
#     neg_dist = ((feat_anchor - feat_negative) ** 2).sum(dim=1)

#     # 指数形式的相似度
#     sim_pos = torch.exp(-pos_dist / (2 * sigma ** 2))
#     sim_neg = torch.exp(-neg_dist / (2 * sigma ** 2))

#     # margin loss
#     loss = torch.clamp(sim_neg - sim_pos + margin, min=0).mean()

#     return loss

def ssl_exponential_contrastive_loss(feat_online_pos, feat_target_pos, feat_target_neg, delta=2.0, sigma=0.3):
    """
    计算自监督对比损失（仅教师网络接收负样本）

    参数说明：
    - feat_online_pos: Tensor, shape (B, D), 学生网络的正样本特征
    - feat_target_pos: Tensor, shape (B, D), 教师网络的正样本特征
    - feat_target_neg: Tensor, shape (B, D), 教师网络的负样本特征
    - delta: float, 缩放因子
    - sigma: float, 平滑参数

    返回：
    - loss: scalar tensor
    """

    # 计算均值平方距离
    def mean_squared_distance(a, b):
        return F.mse_loss(a.mean(dim=0), b.mean(dim=0), reduction='sum')

    # 正样本距离
    pos_dist = mean_squared_distance(feat_online_pos, feat_target_pos)

    # 负样本距离（教师网络内）
    neg_dist = mean_squared_distance(feat_online_pos, feat_target_neg)

    # 按照公式计算损失
    exp_pos = torch.exp(-pos_dist / (2 * sigma**2))
    exp_neg = torch.exp(-neg_dist / (2 * sigma**2))
    loss1 = delta * exp_pos
    loss2 = exp_neg
    loss = loss1 - loss2

    return loss,loss1,loss2

def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    """
    anchor:     shape (B, D)
    positive:   shape (B, D)
    negatives:  shape (B, D) or (B, N, D)
    """
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)

    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)  # (B, 1)

    if negatives.dim() == 2:
        # (B, D) -> (B, 1, D)
        negatives = negatives.unsqueeze(1)
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # (B, N)

    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # (B, 1+N)
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

    loss = F.cross_entropy(logits, labels)
    return loss
    
class Experiment(object):
    def __init__(self, model, model_name, optimizer, train_dataloader,val_dataloader, test_dataloader, model_save_path,data_name,subject,code,prune_lambda,
                 temperature, loss_func, classification, device='cuda:0', epochs=200, num_classes=4,train_dataloader_task2=None, test_dataloader_task2=None):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model_save_path = model_save_path
        self.temperature = temperature
        self.loss_func = loss_func
        self.num_classes=num_classes
        self.classification = classification
        self.device = device
        self.epochs = epochs
        self.num_classes = num_classes
        self.encoded_data = None
        self.writer = None
        self.data_name = data_name                                                                                                                                          
        self.subject = subject
        self.code = code
        self.prune_lambda = prune_lambda
        self.train_dataloader_task2 = train_dataloader_task2
        self.test_dataloader_task2 = test_dataloader_task2    

    # train backbone
                      
    def run(self, data_aug=False):
        log_dir = f"./logs/train_output/{self.model_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.code}_{self.subject}_{self.data_name}_{time.strftime('%Y%m%d_%H-%M-%S')}.txt")
        sys.stdout = Logger(log_file)
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        # for name, param in self.model.named_parameters():   # 冻结 PatchTST 模型
        #     if name.startswith('PatchTST'):
        #         param.requires_grad = False

        iteration = 0
        best_acc = 0
        saved_epoch = 0
        self.writer = SummaryWriter(log_dir=f"./logs/logs_run/{self.model_name}/{self.code}_{self.data_name}")
        for epoch in range(self.epochs):

            print(f"****************************{self.subject}第 {epoch + 1} 轮训练开始****************************")

            # 训练开始
            self.model.train()
            evaluation = []

            for batch_i, (X, y) in enumerate(self.train_dataloader):

                # 1. data augmentation
                if data_aug:
                    origin_X = X
                    trans = []
                    for i in range(origin_X.shape[0]):
                        t1 = transform(origin_X[i], 'permute')
                        trans.append(t1)
                    for i in range(origin_X.shape[0]):
                        t2 = transform(origin_X[i], 'cutout_resize')
                        trans.append(t2)
                    trans = np.reshape(np.concatenate(trans), (-1, origin_X.shape[1], origin_X.shape[2]))
                    trans = torch.tensor(trans, dtype=torch.float).to(self.device)
                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    output = self.model(trans)
                    y = y.to(self.device).long()
                    y = torch.cat((y, y), dim=0)
                # 2. original data
                else:
                    X = X.to(self.device)
                    y = y.to(self.device).long()
                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    if self.model_name in ['PatchTST']:
                        X = X.permute(0, 2, 1)

                    X = X.to(torch.float)
                    output = self.model(X)

                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())
                train_loss = self.loss_func(output, y)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            evaluation = [item for sublist in evaluation for item in sublist]
            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar(f"{self.subject}_loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar(f"{self.subject}_acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch + 1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.val_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device).long()

                    if self.model_name in ['EEG_Conformer', 'LMDANet','EEGNet']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    if self.model_name in ['PatchTST']:
                        X = X.permute(0, 2, 1)
                    X = X.to(torch.float)

                    output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    evaluation.append((predicted == y).tolist())
                    eval_loss = self.loss_func(output, y)

            evaluation = [item for sublist in evaluation for item in sublist]
            eval_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar(f"{self.subject}_loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar(f"{self.subject}_acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch + 1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc
                # if epoch > 50:
                saved_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_save_path)

        # torch.save(self.model.state_dict(), self.model_save_path)
        print(f"\n最佳模型保存在 epoch {saved_epoch}，验证集最佳准确率为: {best_acc:.4f}")

                # ============ 加载测试集并评估模型 ============
        print(f"正在使用最佳模型评估测试集性能（被试 {self.subject}）...")
        # 加载模型权重
        trained_model = load_model(self.model_name, self.model_save_path, self.device)

        # 评估
        _, _, acc = predict_and_evaluate(trained_model, self.test_dataloader, self.device, self.model_name)
        sys.stdout = sys.__stdout__  # 恢复标准输出
        self.writer.add_scalars("final_accuracy", {
            f"{self.subject}_val_acc": best_acc,
            f"{self.subject}_test_acc": acc
        }, global_step=0)
        self.writer.close()
        return train_acc, best_acc

    def run_no_val(self, data_aug=False):
        prune_save_dir = f"./logs/prune_analysis/{self.model_name}"
        os.makedirs(prune_save_dir, exist_ok=True)
        log_dir = f"./logs/train_output/{self.model_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.code}_{self.subject}_{self.data_name}_{time.strftime('%Y%m%d_%H-%M-%S')}.txt")
        prune_save_file = os.path.join(prune_save_dir, f"{self.code}_{self.subject}_{self.data_name}_{time.strftime('%Y%m%d_%H-%M-%S')}")
        os.makedirs(prune_save_file, exist_ok=True)

        sys.stdout = Logger(log_file)

        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        iteration = 0
        best_test_acc = 0
        saved_epoch = 0
        self.writer = SummaryWriter(log_dir=f"./logs/logs_run/{self.model_name}/{self.code}_{self.data_name}")

        for epoch in range(self.epochs):
            print(f"****************************{self.subject} 第 {epoch + 1} 轮训练开始****************************")

            self.model.train()
            evaluation = []

            for batch_i, (X, y) in enumerate(self.train_dataloader):
                if data_aug:
                    origin_X = X
                    trans = []
                    for i in range(origin_X.shape[0]):
                        trans.append(transform(origin_X[i], 'permute'))
                    for i in range(origin_X.shape[0]):
                        trans.append(transform(origin_X[i], 'cutout_resize'))
                    trans = np.reshape(np.concatenate(trans), (-1, origin_X.shape[1], origin_X.shape[2]))
                    X = torch.tensor(trans, dtype=torch.float).to(self.device)
                    y = y.to(self.device).long()
                    y = torch.cat((y, y), dim=0)
                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                else:
                    X = X.to(self.device)
                    y = y.to(self.device).long()
                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    if self.model_name in ['iTransformer', 'TSMixer', 'PatchTST']:
                        X = X.permute(0, 2, 1)

                X = X.to(torch.float)
                output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.extend((predicted == y).tolist())
                train_loss = self.loss_func(output, y)
                sparsity_loss = torch.norm(torch.sigmoid(self.model.A_mask), p=1)  #剪枝loss
                total_loss = train_loss + self.prune_lambda * sparsity_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                iteration += 1

            if (epoch + 1) % (self.epochs//20) == 0:  # 更新一次剪枝结构
                self.model.update_prune_mask()  # 每次剪掉 10% 最弱的连接    
                    # 保存热力图
                adj = self.model.A_pruned.detach().cpu().numpy()
                import matplotlib.pyplot as plt
                plt.figure(figsize=(5, 5))
                plt.imshow(adj, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Subject {self.subject} - Epoch {epoch + 1}')
                plt.savefig(os.path.join(prune_save_file, f'epoch_{epoch+1}.png'))
                plt.close()

                # 保存邻接矩阵值
                np.save(os.path.join(prune_save_file, f'epoch_{epoch+1}_adj.npy'), adj)

            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar(f"{self.subject}_loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar(f"{self.subject}_acc/train_acc", train_acc, iteration)
            scheduler.step()
            print(f"epoch: {epoch + 1}, train_Loss: {train_loss.item():.4f}, train_acc: {train_acc:.4f}")

            # ========= 在测试集上评估，作为“验证” =========
            self.model.eval()
            test_eval = []
            with torch.no_grad():
                for X, y in self.test_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device).long()

                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    if self.model_name in ['PatchTST']:
                        X = X.permute(0, 2, 1)

                    X = X.to(torch.float)
                    output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    test_eval.extend((predicted == y).tolist())
                    test_loss = self.loss_func(output, y)

            test_acc = sum(test_eval) / len(test_eval)
            self.writer.add_scalar(f"{self.subject}_loss/test_loss", test_loss.item(), iteration)
            self.writer.add_scalar(f"{self.subject}_acc/test_acc", test_acc, iteration)
            print(f"epoch: {epoch + 1}, test_Loss: {test_loss.item():.4f}, test_acc: {test_acc:.4f}")

            # 保存在测试集上表现最好的模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                saved_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_save_path)

        
        print(f"\n最佳模型保存在 epoch {saved_epoch}，测试集最佳准确率为: {best_test_acc:.4f}")
        self.writer.add_scalars("final_accuracy", {
            f"{self.subject}_test_acc": best_test_acc
        }, global_step=0)
        self.writer.close()
        # 使用最佳模型做最终评估（结果应与 best_test_acc 一致）
        print(f"正在使用最佳模型评估测试集性能（被试 {self.subject}）...")
        trained_model = load_model(self.model_name, self.model_save_path, self.device)
        _, _, final_test_acc = predict_and_evaluate(trained_model, self.test_dataloader, self.device, self.model_name)

        sys.stdout = sys.__stdout__  # 恢复标准输出
        np.save(os.path.join(prune_save_file, 'final_adj.npy'), self.model.A_pruned.detach().cpu().numpy())

        return train_acc, final_test_acc


    def run_no_val_EEG(self, data_aug=False):
        log_dir = f"./logs/train_output/{self.model_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.code}_{self.subject}_{self.data_name}_{time.strftime('%Y%m%d_%H-%M-%S')}.txt")
        self.log_file = log_file
        sys.stdout = Logger(log_file)

        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        iteration = 0
        best_test_acc = 0
        saved_epoch = 0
        self.writer = SummaryWriter(log_dir=f"./logs/logs_run/{self.model_name}/{self.code}_{self.data_name}")
        summary_log_path  = f"./logs/result/{self.model_name}/{self.code}_{self.data_name}.txt"
        os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)  # 确保目录存在
        # summary_log_path = os.path.join(summary_dir, "summary_result.txt")

        for epoch in range(self.epochs):
            print(f"****************************{self.subject} 第 {epoch + 1} 轮训练开始****************************")

            self.model.train()
            evaluation = []

            for batch_i, (X, y) in enumerate(self.train_dataloader):
                if data_aug:
                    origin_X = X
                    trans = []
                    for i in range(origin_X.shape[0]):
                        trans.append(transform(origin_X[i], 'permute'))
                    for i in range(origin_X.shape[0]):
                        trans.append(transform(origin_X[i], 'cutout_resize'))
                    trans = np.reshape(np.concatenate(trans), (-1, origin_X.shape[1], origin_X.shape[2]))
                    X = torch.tensor(trans, dtype=torch.float).to(self.device)
                    y = y.to(self.device).long()
                    y = torch.cat((y, y), dim=0)
                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                else:
                    X = X.to(self.device)
                    y = y.to(self.device).long()
                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    if self.model_name in ['iTransformer', 'TSMixer', 'PatchTST']:
                        X = X.permute(0, 2, 1)

                X = X.to(torch.float)
                output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.extend((predicted == y).tolist())
                train_loss = self.loss_func(output, y)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1


            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar(f"{self.subject}_loss/train_loss", train_loss.item(), epoch)
            self.writer.add_scalar(f"{self.subject}_acc/train_acc", train_acc, epoch)
            scheduler.step()
            print(f"epoch: {epoch + 1}, train_Loss: {train_loss.item():.4f}, train_acc: {train_acc:.4f}")

            # ========= 在测试集上评估，作为“验证” =========
            self.model.eval()
            test_eval = []
            with torch.no_grad():
                for X, y in self.test_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device).long()

                    if self.model_name in ['EEG_Conformer', 'LMDANet', 'EEGNet','LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    if self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    if self.model_name in ['PatchTST']:
                        X = X.permute(0, 2, 1)

                    X = X.to(torch.float)
                    output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    test_eval.extend((predicted == y).tolist())
                    test_loss = self.loss_func(output, y)

            test_acc = sum(test_eval) / len(test_eval)
            self.writer.add_scalar(f"{self.subject}_loss/test_loss", test_loss.item(), epoch)
            self.writer.add_scalar(f"{self.subject}_acc/test_acc", test_acc, epoch)
            print(f"epoch: {epoch + 1}, test_Loss: {test_loss.item():.4f}, test_acc: {test_acc:.4f}")
            # if (epoch + 1) % 100 == 0:
            #     print(f"正在生成第 {epoch + 1} 轮的 t-SNE 可视化图...")
            #     self.plot_tsne(self.model, self.test_dataloader, epoch + 1)
            # 保存在测试集上表现最好的模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                saved_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_save_path)

        print(f"\n最佳模型保存在 epoch {saved_epoch}，测试集最佳准确率为: {best_test_acc:.4f}")
                    
        # print(f"正在生成第 {epoch + 1} 轮的 t-SNE 可视化图...")
        # self.plot_tsne(self.model, self.test_dataloader, epoch + 1)
        self.writer.add_scalars("final_accuracy", {
            f"{self.subject}_test_acc": best_test_acc
        }, global_step=0)

        self.writer.close()
        # 使用最佳模型做最终评估（结果应与 best_test_acc 一致）
        print(f"正在使用最佳模型评估测试集性能（被试 {self.subject}）...")
        trained_model = load_model(self.model_name, self.model_save_path,self.num_classes, self.device)
        _, _, final_test_acc = predict_and_evaluate(trained_model, self.test_dataloader, self.device, self.model_name)

        with open(summary_log_path, "a") as f:
            f.write(f"Subject: {self.subject} | Final Train Acc: {train_acc:.4f} | Final Test Acc: {test_acc:.4f} | Best_test_acc: {best_test_acc:.4f}| Best Epoch: {saved_epoch}\n")
        
        sys.stdout = sys.__stdout__  # 恢复标准输出

        return train_acc, final_test_acc,best_test_acc
    
    
    def run_contrastive_finetune_prescreening(self, pos_loader, neg_loader, test_loader, delta=2.0, sigma=0.3, epochs=100, ema_decay=0.9995):
        sys.stdout = Logger(self.log_file)
        
        print(f" 开始对被试 {self.subject} 进行自监督微调...")
        if not hasattr(self, 'writer'):
            self.writer = SummaryWriter(log_dir=f"./logs/logs_run/{self.model_name}/{self.code}_{self.data_name}")

        # 显式加载已训练好的模型权重
        print(f"正在加载已训练的模型权重: {self.model_save_path}")
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        # 初始化 f_φ = f_θ (teacher = student)
        teacher_model = copy.deepcopy(self.model)
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        iteration = 0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        best_acc = 0.0
        best_epoch = 0
        log_path = f"./logs/result/{self.model_name}/{self.code}_{self.data_name}.txt"

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for (pos_data, _), (neg_data, _) in zip(pos_loader, neg_loader):
                pos_data = pos_data.to(self.device).float()
                neg_data = neg_data.to(self.device).float()

                if self.model_name in ['EEGNet', 'LMDANet', 'EEG_Conformer', 'LMDANet_GCN']:
                    pos_data = pos_data.unsqueeze(1)
                    neg_data = neg_data.unsqueeze(1)
                elif self.model_name in ['dfformer']:
                    pos_data = pos_data.unsqueeze(2)
                    neg_data = neg_data.unsqueeze(2)

                feat_online_pos = self.model(pos_data, return_feature=True)
                feat_target_pos = teacher_model(pos_data, return_feature=True)
                feat_target_neg = teacher_model(neg_data, return_feature=True)

                # loss = ssl_exponential_contrastive_loss(
                #     feat_online_pos, feat_target_pos, feat_target_neg,
                #     delta=delta, sigma=sigma
                # )                
                loss,sim_neg,sim_pos = ssl_exponential_contrastive_loss(
                    feat_online_pos, feat_target_pos, feat_target_neg,
                    sigma=sigma
                )                
                # loss = info_nce_loss(
                #     feat_online_pos, feat_target_pos, feat_target_neg
                # )
            # neg_iter = iter(neg_loader)  # 构造 neg_loader 的迭代器
            # neg_all = list(neg_loader)  # 每个 epoch 拷贝一份
            # random.shuffle(neg_all)
            # neg_iter = iter(neg_all)
            # for (pos_data, _) in pos_loader:
            #     pos_data = pos_data.to(self.device).float()

            #     # 获取 5 个负样本 batch
            #     neg_feat_list = []
            #     for _ in range(10):
            #         try:
            #             neg_data, _ = next(neg_iter)
            #         except StopIteration:
            #             # 如果迭代完了，重新来一次
            #             neg_iter = iter(neg_loader)
            #             neg_data, _ = next(neg_iter)
                    
            #         neg_data = neg_data.to(self.device).float()
                    
            #         if self.model_name in ['EEGNet', 'LMDANet', 'EEG_Conformer', 'LMDANet_GCN']:
            #             neg_data = neg_data.unsqueeze(1)
            #         elif self.model_name in ['dfformer']:
            #             neg_data = neg_data.unsqueeze(2)

            #         neg_feat = teacher_model(neg_data, return_feature=True)  # (B, D)
            #         neg_feat_list.append(neg_feat.unsqueeze(1))  # (B, 1, D)

            #     feat_target_neg = torch.cat(neg_feat_list, dim=1)  # (B, 5, D)

            #     # 正样本处理
            #     if self.model_name in ['EEGNet', 'LMDANet', 'EEG_Conformer', 'LMDANet_GCN']:
            #         pos_data = pos_data.unsqueeze(1)
            #     elif self.model_name in ['dfformer']:
            #         pos_data = pos_data.unsqueeze(2)

            #     feat_online_pos = self.model(pos_data, return_feature=True)   # (B, D)
            #     feat_target_pos = teacher_model(pos_data, return_feature=True)  # (B, D)

            #     # InfoNCE 损失（多负样本）
            #     loss = info_nce_loss(
            #         feat_online_pos,         # anchor
            #         feat_target_pos,         # positive
            #         feat_target_neg          # (B, 5, D)
            #     )
                self.writer.add_scalar(f"{self.subject}_loss/sim_neg_loss", sim_neg.item(), iteration)
                self.writer.add_scalar(f"{self.subject}_loss/sim_pos_loss", sim_pos.item(), iteration)
                self.writer.add_scalar(f"{self.subject}_loss/sum_loss", loss.item(), epoch)
                
                iteration += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for param_q, param_k in zip(self.model.parameters(), teacher_model.parameters()):
                        param_k.data.mul_(ema_decay).add_(param_q.data, alpha=1.0 - ema_decay)

                epoch_loss += loss.item()
                num_batches += 1

            # 测试集评估
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(self.device).float(), y.to(self.device).long()
                    if self.model_name in ['EEGNet', 'LMDANet', 'EEG_Conformer', 'LMDANet_GCN']:
                        X = X.unsqueeze(1)
                    elif self.model_name in ['dfformer']:
                        X = X.unsqueeze(2)
                    output = self.model(X)
                    _, pred = output.max(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_save_path)

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss / num_batches:.6f} | Test Acc: {acc:.4f} {'(best)' if is_best else ''}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"正在生成第 {epoch + 1} 轮的 t-SNE 可视化图...")
                self.plot_tsne(self.model, test_loader, epoch + 1)
        with open(log_path, 'a') as f:
            f.write(f"Subject: {self.subject} | Final Test Acc: {best_acc:.4f} | Best Epoch: {best_epoch}\n")

        print(f"微调完成！最佳准确率：{best_acc:.4f}（在第 {best_epoch} 轮）")

    def plot_tsne(self, model, dataloader, epoch,task =None):
        model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device).float()
                y = y.to(self.device).long()

                if self.model_name in ['EEGNet', 'LMDANet',  'LMDA_Multi_Task', 'LMDANet_GCN']:
                    X = X.unsqueeze(1)
                elif self.model_name in ['dfformer']:
                    X = X.unsqueeze(2)

                features = model(X, return_feature=True)
                all_features.append(features.cpu())
                all_labels.append(y.cpu())

        all_features = torch.cat(all_features, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
        all_features = all_features.reshape(all_features.shape[0], -1)
        tsne_result = tsne.fit_transform(all_features)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=all_labels, palette="tab10", legend='full')
        plt.title(f"t-SNE Visualization at Epoch {epoch}")
        os.makedirs(f"./logs/tsne/{self.model_name}", exist_ok=True)
        if task == None:
            save_path = f"./logs/tsne/{self.model_name}/{self.code}_{self.subject}_{self.data_name}_epoch{epoch}.png"
        else:
            save_path = f"./logs/tsne/{self.model_name}/{self.code}_{self.subject}_{self.data_name}_{task}_epoch{epoch}.png"

        plt.savefig(save_path)
        plt.close()
        print(f"t-SNE 可视化已保存至: {save_path}")

    def run_Multi_Task(self, data_aug=False):
        log_dir = f"./logs/train_output/{self.model_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.code}_{self.subject}_{self.data_name}_{time.strftime('%Y%m%d_%H-%M-%S')}.txt")
        self.log_file = log_file
        sys.stdout = Logger(log_file)

        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)
        best_test_acc = 0
        saved_epoch = 0
        self.writer = SummaryWriter(log_dir=f"./logs/logs_run/{self.model_name}/{self.code}_{self.data_name}")
        summary_log_path = f"./logs/result/{self.model_name}/{self.code}_{self.data_name}.txt"
        os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)

        for epoch in range(self.epochs):
            print(f"********** Subject {self.subject} - Epoch {epoch + 1} **********")
            self.model.train()
            # self.freeze_task2_parts()  # 冻结task2相关部分
            total_loss_epoch = 0
            total_acc_task1 = []
            total_acc_task2 = []

            # ============ Task 1 - 二分类训练 ============
            for X, y1 in self.train_dataloader:
                if self.model_name in ['LMDA_Multi_Task']:
                    X = X.to(self.device).unsqueeze(1).float()
                else:
                    X = X.to(self.device).float()
                y1 = y1.to(self.device).long()

                features = self.model.backbone(X)
                logits1 = self.model.classifier1(features)
                loss1 = self.loss_func(logits1, y1)

                self.optimizer.zero_grad()
                loss1.backward()
                # 冻结 classifier2 的参数不更新
                self.model.classifier2.requires_grad_(False)
                self.optimizer.step()
                self.model.classifier2.requires_grad_(True)

                pred1 = torch.argmax(logits1, dim=1)
                total_acc_task1.extend((pred1 == y1).cpu().tolist())
                total_loss_epoch += loss1.item()
                # self.unfreeze_all()  # 恢复所有参数参与训练
            # ============ Task 2 - 四分类训练 ============
            for X, y2 in self.train_dataloader_task2:
                if self.model_name in ['LMDA_Multi_Task']:
                    X = X.to(self.device).unsqueeze(1).float()
                else:
                    X = X.to(self.device).float()                    
                y2 = y2.to(self.device).long()

                features = self.model.backbone(X)
                logits2 = self.model.classifier2(features)
                loss2 = self.loss_func(logits2, y2)

                self.optimizer.zero_grad()
                loss2.backward()
                # 冻结 classifier1 的参数不更新
                self.model.classifier1.requires_grad_(False)
                self.optimizer.step()
                self.model.classifier1.requires_grad_(True)

                pred2 = torch.argmax(logits2, dim=1)
                total_acc_task2.extend((pred2 == y2).cpu().tolist())
                total_loss_epoch += loss2.item()

            # ===== Logging =====
            acc1 = sum(total_acc_task1) / len(total_acc_task1)
            acc2 = sum(total_acc_task2) / len(total_acc_task2)
            self.writer.add_scalar(f"{self.subject}_loss/train_loss", total_loss_epoch, epoch)
            self.writer.add_scalar(f"{self.subject}_acc/train_acc_task1", acc1, epoch)
            self.writer.add_scalar(f"{self.subject}_acc/train_acc_task2", acc2, epoch)
            print(f"Train Loss={total_loss_epoch:.4f}")
            print(f"Task1 Train Acc={acc1:.4f}, Task2 Train Acc={acc2:.4f}")

            scheduler.step()

            # ========== 验证阶段 ==========
            self.model.eval()
            test_eval_task1 = []
            test_eval_task2 = []
            total_test_loss_task1 = 0.0
            total_test_loss_task2 = 0.0

            with torch.no_grad():
                for X, y1 in self.test_dataloader:
                    if self.model_name in ['LMDA_Multi_Task']:
                        X = X.to(self.device).unsqueeze(1).float()
                    else:
                        X = X.to(self.device).float()                        
                    y1 = y1.to(self.device).long()
                    features = self.model.backbone(X)
                    logits1 = self.model.classifier1(features)
                    loss1 = self.loss_func(logits1, y1)
                    pred1 = torch.argmax(logits1, dim=1)
                    total_test_loss_task1 += loss1.item()
                    test_eval_task1.extend((pred1 == y1).cpu().tolist())

                for X, y2 in self.test_dataloader_task2:
                    if self.model_name in ['LMDA_Multi_Task']:
                        X = X.to(self.device).unsqueeze(1).float()
                    else:
                        X = X.to(self.device).float()                                            
                    y2 = y2.to(self.device).long()
                    features = self.model.backbone(X)
                    logits2 = self.model.classifier2(features)
                    loss2 = self.loss_func(logits2, y2)
                    pred2 = torch.argmax(logits2, dim=1)
                    total_test_loss_task2 += loss2.item()
                    test_eval_task2.extend((pred2 == y2).cpu().tolist())

            test_acc_task1 = sum(test_eval_task1) / len(test_eval_task1)
            test_acc_task2 = sum(test_eval_task2) / len(test_eval_task2)
            self.writer.add_scalar(f"{self.subject}_loss/test_loss_task1", total_test_loss_task1, epoch)
            self.writer.add_scalar(f"{self.subject}_loss/test_loss_task2", total_test_loss_task2, epoch)
            self.writer.add_scalar(f"{self.subject}_acc/test_acc_task1", test_acc_task1, epoch)
            self.writer.add_scalar(f"{self.subject}_acc/test_acc_task2", test_acc_task2, epoch)
            print(f"Task1 Test Acc={test_acc_task1:.4f}, Task2 Test Acc={test_acc_task2:.4f}")

            if test_acc_task2 > best_test_acc:
                best_test_acc = test_acc_task2
                saved_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_save_path)

        print(f"\nBest Test Accuracy: {best_test_acc:.4f} at epoch {saved_epoch}")
        self.writer.close()
        sys.stdout = sys.__stdout__
        trained_model = load_model(self.model_name, self.model_save_path, self.num_classes, self.device)
        # _, _, final_test_acc = predict_and_evaluate_multi_task(trained_model, self.test_dataloader, self.device, self.model_name)
        _, _, acc_task1 = predict_and_evaluate_multi_task(trained_model, self.test_dataloader, self.device, task=1)
        _, _, acc_task2 = predict_and_evaluate_multi_task(trained_model, self.test_dataloader_task2, self.device, task=2)
        self.plot_tsne(self.model, self.test_dataloader, epoch + 1,task=1)
        self.plot_tsne(self.model, self.test_dataloader_task2, epoch + 1,task=2)

        with open(summary_log_path, "a") as f:
            f.write(
                f"Subject: {self.subject} | "
                f"Train Task1 Acc: {acc1:.4f} | Train Task2 Acc: {acc2:.4f} | "
                f"Final Test Task1 Acc: {test_acc_task1:.4f} | Final Test Task2 Acc: {test_acc_task2:.4f} | "
                f"Best Task1 Test Acc: {best_test_acc:.4f} | Best Epoch: {saved_epoch}\n"
            )
        return acc1, acc_task1,acc_task2, best_test_acc

    def freeze_task2_parts(self):
        for name, param in self.model.backbone.named_parameters():
            if 'features1' in name or 'cbam1' in name or 'features2' in name or 'cbam2' in name:
                param.requires_grad = False
        for param in self.model.classifier2.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
