import os
import shutil
# from umap import UMAP
import sys
import time

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

from data_processing.data_augmentation import Transform, trans_example_plot

sys.path.append("/home/work3/wkh/CL-Model")


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


class Experiment(object):
    def __init__(self, model, model_name, optimizer, train_dataloader, test_dataloader, model_save_path,
                 temperature, loss_func, classification, device='cuda:0', epochs=200, num_classes=4):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model_save_path = model_save_path
        self.temperature = temperature
        self.loss_func = loss_func
        self.classification = classification
        self.device = device
        self.epochs = epochs
        self.num_classes = num_classes
        self.encoded_data = None
        self.writer = None

    # train backbone

    def run(self, data_aug=False):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        # for name, param in self.model.named_parameters():   # 冻结 PatchTST 模型
        #     if name.startswith('PatchTST'):
        #         param.requires_grad = False

        iteration = 0
        best_acc = 0
        saved_epoch = 0
        self.writer = SummaryWriter(log_dir=f"./logs_run/{self.model_name}_{time.strftime('%Y%m%d_%H-%M-%S')}")
        for epoch in range(self.epochs):

            print("****************************第 {} 轮训练开始****************************".format(epoch + 1))

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
            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar("acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch + 1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
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
            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar("acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch + 1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc
                # if epoch > 50:
                saved_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_save_path)

        self.writer.close()
        # torch.save(self.model.state_dict(), self.model_save_path)
        print("best acc: {}".format(best_acc))
        print("save model epoch: {}".format(saved_epoch))

        return train_acc, best_acc
    def train_run(self):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        iteration = 0
        train_losses = []
        eval_losses = []
        self.writer = SummaryWriter(log_dir=f"./logs_run/{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}")
        self.writer = SummaryWriter("./logs_train")
        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

            # 训练开始
            self.model.train()
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                # printProgressBar(batch_i, train_dataloader.__len__())
                X = X.to(self.device)
                y = y.to(self.device)

                trans = []
                for i in range(X.shape[0]):
                    t1 = transform(X[i], 'permute')
                    trans.append(t1)
                for i in range(X.shape[0]):
                    t2 = transform(X[i], 'cutout_resize')
                    trans.append(t2)
                trans = np.reshape(np.concatenate(trans), (-1, X.shape[1], X.shape[2]))
                trans = torch.tensor(trans, dtype=torch.float, device="cuda")

                # 用于自注意力机制
                if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                    trans = trans.unsqueeze(1)
                if self.model_name in ['PatchTST']:
                    trans = trans.permute(0, 2, 1)
                output = self.model(trans)

                train_loss = contrast_loss(output, temperature=self.temperature)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}".format(epoch+1, train_loss.item()))
            train_losses.append(train_loss.item())

            # 测试步骤开始
            self.model.eval()
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    trans = []
                    for i in range(X.shape[0]):
                        t1 = transform(X[i], 'permute')
                        trans.append(t1)
                    for i in range(X.shape[0]):
                        t2 = transform(X[i], 'cutout_resize')
                        trans.append(t2)
                    trans = np.reshape(np.concatenate(trans), (-1, X.shape[1], X.shape[2]))
                    trans = torch.tensor(trans, dtype=torch.float, device="cuda")

                    # 用于自注意力机制
                    if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                        trans = trans.unsqueeze(1)
                    output = self.model(trans)
                    eval_loss = contrast_loss(output, temperature=self.temperature)

            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            print("epoch: {}, eval_Loss: {}".format(epoch+1, eval_loss.item()))
            eval_losses.append(eval_loss.item())

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)
        # return train_losses

    # model fine-tune
    def train_run_finetune(self):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        self.model.load_state_dict(torch.load(self.model_save_path))

        iteration = 0
        best_acc = 0
        shutil.rmtree("./logs_fine-tune")
        if os.path.exists("./logs_draw_tsne"):
            shutil.rmtree("./logs_draw_tsne")
        os.makedirs("./logs_draw_tsne", exist_ok=True)
        self.writer = SummaryWriter("./logs_fine-tune")
        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

            # 训练开始
            self.model.train()
            evaluation = []
            self.encoded_data = []
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                # printProgressBar(batch_i, train_dataloader.__len__())
                X = X.to(self.device)
                y = y.to(self.device).long()

                if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                    X = X.unsqueeze(1)
                X = X.to(torch.float)

                # fine-tune
                output = self.model(X)
                # self.encoded_data.append(output.cpu().squeeze().detach().numpy())
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())
                train_loss = self.loss_func(output, y)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            # self.encoded_data = np.concatenate(self.encoded_data)
            evaluation = [item for sublist in evaluation for item in sublist]
            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar("acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch+1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device).long()

                    if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                        X = X.unsqueeze(1)
                    X = X.to(torch.float)

                    output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    evaluation.append((predicted == y).tolist())
                    eval_loss = self.loss_func(output, y)

            evaluation = [item for sublist in evaluation for item in sublist]
            eval_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar("acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch+1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path[:-4] + "_classification.pth")
        print("best acc: {}".format(best_acc))
        # return train_losses




    def run_card(self, data_aug=False):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        iteration = 0
        best_acc = 0
        shutil.rmtree("./logs_train")
        self.writer = SummaryWriter("./logs_train")
        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

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
                    if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer', 'card']:
                        trans = trans.unsqueeze(1)
                    trans = trans.permute(0, 2, 1)
                    output = self.model(trans)
                    y = y.to(self.device).long()
                    y = torch.cat((y, y), dim=0)
                # 2. original data
                else:
                    X = X.to(self.device)
                    y = y.to(self.device).long()
                    if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer', 'card']:
                        X = X.unsqueeze(1)
                    X = X.to(torch.float)
                    X = X.permute(0, 2, 1)          # [64, 22, 1000] -> [64, 1000, 22]
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
            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar("acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch+1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device).long()

                    if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer', 'card']:
                        X = X.unsqueeze(1)
                    X = X.to(torch.float)
                    X = X.permute(0, 2, 1)

                    output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    evaluation.append((predicted == y).tolist())
                    eval_loss = self.loss_func(output, y)

            evaluation = [item for sublist in evaluation for item in sublist]
            eval_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar("acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch+1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)
        print("best acc: {}".format(best_acc))

        return train_acc, best_acc


    def run_add_loss_contrast(self, lamda):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        # for name, param in self.model.named_parameters():   # 冻结 PatchTST 模型
        #     if name.startswith('PatchTST'):
        #         param.requires_grad = False

        iteration = 0
        best_acc = 0
        shutil.rmtree("./logs_run")
        self.writer = SummaryWriter("./logs_run")
        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

            # 训练开始
            self.model.train()
            evaluation = []
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                # printProgressBar(batch_i, train_dataloader.__len__())
                X = X.to(self.device)
                y = y.to(self.device).long()
                X = X.to(torch.float)

                # 计算标签损失 label_loss
                embed, output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())
                label_loss = self.loss_func(output, y)

                trans = []
                for i in range(X.shape[0]):
                    t1 = transform(X[i], 'permute')
                    trans.append(t1)
                for i in range(X.shape[0]):
                    t2 = transform(X[i], 'cutout_resize')
                    trans.append(t2)
                trans = np.reshape(np.concatenate(trans), (-1, X.shape[1], X.shape[2]))
                trans = torch.tensor(trans, dtype=torch.float).to(self.device)
                embed, output = self.model(trans)
                contrastive_loss = contrast_loss(embed, temperature=self.temperature)

                train_loss = lamda * label_loss + (1 - lamda) * contrastive_loss
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            evaluation = [item for sublist in evaluation for item in sublist]
            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar("acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch+1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device).long()
                    X = X.to(torch.float)

                    # 计算标签损失 label_loss
                    embed, output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    evaluation.append((predicted == y).tolist())
                    label_loss = self.loss_func(output, y)

                    trans = []
                    for i in range(X.shape[0]):
                        t1 = transform(X[i], 'permute')
                        trans.append(t1)
                    for i in range(X.shape[0]):
                        t2 = transform(X[i], 'cutout_resize')
                        trans.append(t2)
                    trans = np.reshape(np.concatenate(trans), (-1, X.shape[1], X.shape[2]))
                    trans = torch.tensor(trans, dtype=torch.float, device="cuda")
                    embed, output = self.model(trans)
                    contrastive_loss = contrast_loss(embed, temperature=self.temperature)

                    eval_loss = lamda * label_loss + (1 - lamda) * contrastive_loss

            evaluation = [item for sublist in evaluation for item in sublist]
            eval_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar("acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch+1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)
        print("best acc: {}".format(best_acc))

    def run_fine_tone_PatchTST_mse(self):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        iteration = 0
        shutil.rmtree("./logs_mse")
        self.writer = SummaryWriter("./logs_mse")

        criterion = nn.MSELoss()
        self.model.load_state_dict(torch.load('model/checkpoint.pth'))
        saved_for_1 = False
        saved_for_05 = False
        saved_for_01 = False

        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

            # 训练开始
            self.model.train()
            evaluation = []
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                # printProgressBar(batch_i, train_dataloader.__len__())
                X = X.to(self.device)
                y = y.to(self.device).long()

                X = X.to(torch.float)
                X = X.permute(0, 2, 1)
                front, rear = torch.split(X, 500, dim=1)    # [64, 500, 22]

                output = self.model(front)
                train_loss = criterion(output, rear)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}".format(epoch+1, train_loss.item()))

            # 根据第一次达到的 mse 损失分级保存模型
            if train_loss.item() < 1 and not saved_for_1:
                torch.save(self.model.state_dict(), self.model_save_path[:-4] + '_model_mse_less_than_1.pth')
                saved_for_1 = True
                print(f"Model saved with MSE < 1 at Epoch {epoch+1}")

            if train_loss.item() < 0.5 and not saved_for_05:
                torch.save(self.model.state_dict(), self.model_save_path[:-4] + '_model_mse_less_than_05.pth')
                saved_for_05 = True
                print(f"Model saved with MSE < 0.5 at Epoch {epoch+1}")

            if train_loss.item() < 0.1 and not saved_for_01:
                torch.save(self.model.state_dict(), self.model_save_path[:-4] + '_model_mse_less_than_01.pth')
                saved_for_01 = True
                print(f"Model saved with MSE < 0.1 at Epoch {epoch+1}")

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device).long()

                    X = X.to(torch.float)
                    X = X.permute(0, 2, 1)
                    front, rear = torch.split(X, 500, dim=1)    # [64, 500, 22]

                    output = self.model(front)
                    eval_loss = criterion(output, rear)

            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            print("epoch: {}, eval_Loss: {}".format(epoch+1, eval_loss.item()))

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)

    # each 10 epochs draw a t-sne
    def draw_running_feature(self, epoch):
        encoded_data = self.encoded_data
        labels = self.train_dataloader.dataset.labels
        labels = labels.detach().numpy()
        mapping = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}
        labels = np.vectorize(mapping.get)(labels)

        tsne = TSNE(n_components=2, random_state=42)
        result = tsne.fit(encoded_data)

        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        result = (result - mean) / std

        color_map = {'Left': 'r', 'Right': 'b', 'Foot': 'c', 'Tongue': 'g'}
        for i, label in enumerate(labels):
            plt.scatter(result[i, 0], result[i, 1], color=color_map[label])

        # 设置图例
        for label, color in color_map.items():
            plt.scatter([], [], color=color, label=label)
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('A01-A08 t-SNE train_sets epoch:{}'.format(epoch))  # PCA t-SNE  train_sets test_sets

        save_path = "./logs_draw_tsne/"
        file_name = "plot_epoch_{}.png".format(epoch)
        save_file = save_path + file_name

        plt.savefig(save_file)
        plt.close()

    # draw embedding feature
    def draw_feature(self, method="t-SNE", test_draw=False, point_num=288):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        dataloader_dataset = self.train_dataloader
        # 创建一个字典映射，使用字典映射替换数组中的值
        mapping = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}

        encoded_data = []
        target_data = []
        for data, target in dataloader_dataset:
            data = torch.Tensor(data)
            data = data.to(self.device)

            data = data.float()
            if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                data = data.unsqueeze(1)
            # 获取features
            output = self.model(data)
            encoded_data.append(output.cpu().squeeze().detach().numpy())
            target_data.append(target.cpu().squeeze().detach().numpy())
        encoded_data = np.concatenate(encoded_data)
        target_data = np.concatenate(target_data)
        labels = np.vectorize(mapping.get)(target_data)
        print(encoded_data.shape)

        if method == "PCA":
            pca = PCA(n_components=2, random_state=42)
            result = pca.fit_transform(encoded_data)
        elif method == "t-SNE":
            tsne = TSNE(n_components=2, random_state=42)
            result = tsne.fit(encoded_data)
        else:
            return False

        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        result = (result - mean) / std
        print(result.shape)

        color_map = {'Left': 'r', 'Right': 'b', 'Foot': 'c', 'Tongue': 'g'}
        num_p = 0
        for i, label in enumerate(labels):
            if num_p < point_num:
                plt.scatter(result[i, 0], result[i, 1], color=color_map[label])
                num_p += 1
            else:
                break

        # 设置图例
        for label, color in color_map.items():
            plt.scatter([], [], color=color, label=label)
        plt.legend()

        # ==============把验证集数据放入训练集的特征空间中======================
        if test_draw:
            dataloader_test = self.test_dataloader
            mapping = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}

            encoded_test = []
            target_test = []
            for data, target in dataloader_test:
                data = torch.Tensor(data)
                data = data.to(self.device)
                data = data.float()

                if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                    data = data.unsqueeze(1)
                output = self.model(data)
                encoded_test.append(output.cpu().squeeze().detach().numpy())
                target_test.append(target.cpu().squeeze().detach().numpy())
            encoded_test = np.concatenate(encoded_test)
            target_test = np.concatenate(target_test)
            test_labels = np.vectorize(mapping.get)(target_test)
            if method == "t-SNE":
                model = TSNE().fit(encoded_data)
            elif method == "UMAP":
                model = UMAP().fit(encoded_data)
            else:
                return
            embedding_test = model.transform(encoded_test)
            embedding_test = (embedding_test - mean) / std

            for i, label in enumerate(test_labels):
                plt.scatter(embedding_test[i, 0], embedding_test[i, 1], edgecolors=color_map[label], facecolor='none')
        # =================================================================

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('t-SNE Result')  # PCA t-SNE UMAP train_sets test_sets
        plt.show()

    def test_run_classification(self):
        # load model
        self.model.load_state_dict(torch.load(self.model_save_path))

        self.model.eval()
        evaluation = []
        with torch.no_grad():
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device).long()

                if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                    X = X.unsqueeze(1)
                X = X.to(torch.float)

                output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())

            for batch_i, (X, y) in enumerate(self.test_dataloader):
                X = X.to(self.device)
                y = y.to(self.device).long()
                
                if self.model_name not in ['PatchTST', 'iTransformer', 'MixFormer']:
                    X = X.unsqueeze(1)
                X = X.to(torch.float)

                output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())

        evaluation = [item for sublist in evaluation for item in sublist]
        eval_acc = sum(evaluation) / len(evaluation)
        print("eval_acc: {}".format(eval_acc))

    def draw_point_feature(self, train_data, point_num=288):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        dataloader_dataset = self.train_dataloader
        # 创建一个字典映射，使用字典映射替换数组中的值
        mapping = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}

        encoded_data = []
        target_data = []
        for data, target in dataloader_dataset:
            data = torch.Tensor(data)
            data = data.to(self.device)

            data = data.float()
            data = data.unsqueeze(1)
            # 获取features
            output = self.model(data)
            encoded_data.append(output.cpu().squeeze().detach().numpy())
            target_data.append(target.cpu().squeeze().detach().numpy())
        encoded_data = np.concatenate(encoded_data)
        target_data = np.concatenate(target_data)
        origin_labels = np.vectorize(mapping.get)(target_data)
        print(encoded_data.shape)

        # 使用t-SNE进行降维
        # tsne = TSNE(n_components=2, random_state=42)
        # result = tsne.fit(encoded_data)
        umap = UMAP(n_components=2, random_state=42, n_neighbors=72)  # metric='cosine'
        result = umap.fit_transform(encoded_data)

        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        result = (result - mean) / std
        print(result.shape)

        color_map = {'Left': 'r', 'Right': 'b', 'Foot': 'c', 'Tongue': 'g'}

        draw_data = []
        draw_label = []
        for data, target in train_data:
            data = torch.Tensor(data)
            data = data.to(self.device)

            data = data.float()
            data = data.unsqueeze(1)
            # 获取features
            output = self.model(data)
            draw_data.append(output.cpu().squeeze().detach().numpy())
            draw_label.append(target.cpu().squeeze().detach().numpy())
        draw_data = np.concatenate(draw_data)
        draw_label = np.concatenate(draw_label)
        draw_label = np.vectorize(mapping.get)(draw_label)

        # 将点投影至嵌入特征空间
        # tsne1 = TSNE().fit(encoded_data)
        # draw_data = tsne1.transform(draw_data)
        umap1 = UMAP().fit(encoded_data)
        draw_data = umap1.transform(draw_data)
        draw_data = (draw_data - mean) / std

        # 创建一个画布和坐标轴
        fig, ax = plt.subplots()
        # 用于绘制新点的对象
        new_point, = ax.plot([], [], marker='*', markersize=20, markerfacecolor='none')

        # 绘制已有的点
        num_p = 0
        for i, label in enumerate(origin_labels):
            if num_p < point_num:
                ax.scatter(result[i, 0], result[i, 1], color=color_map[label])
                num_p += 1
            else:
                break
        # 设置图例
        for label, color in color_map.items():
            ax.scatter([], [], color=color, label=label)
        ax.legend()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('UMAP Result')

        # 动画更新函数
        def update(frame):
            if frame < len(draw_data):
                new_x, new_y = draw_data[frame, 0], draw_data[frame, 1]
                new_point.set_data(new_x, new_y)
                new_point.set_markeredgecolor(color_map[draw_label[frame]])

                # 更新坐标轴范围
                all_x = np.append(result[:num_p, 0], new_x)
                all_y = np.append(result[:num_p, 1], new_y)
                ax.set_xlim(all_x.min() - 0.3, all_x.max() + 0.3)
                ax.set_ylim(all_y.min() - 0.3, all_y.max() + 0.3)

                print(draw_label[frame])
            return new_point,

        # 设置动画
        ani = FuncAnimation(fig, update, frames=range(len(encoded_data)), interval=4000, blit=True, repeat=False)

        # 展示图表
        plt.show()

    # -----------------------------------------------------------------------mask--------------------------------------------------------------------------------

    def mask_even_patches(self, eeg_signal):
        """
        Mask the even patches of an EEG signal.
        """
        # Create a mask for even patches
        mask = torch.arange(eeg_signal.size(1)) % 2 == 0
        mask = mask[None, :, None, None].expand_as(eeg_signal)

        # Apply the mask to the EEG signal
        eeg_signal[mask] = 0
        return eeg_signal
    
    def mask_random(self, eeg_signal, mask_radio=0.5):
        """
        Mask approximately 50% of the EEG signal patches randomly.
        """
        # Create a random mask for approximately half of the patches
        random_mask = torch.rand(eeg_signal.size(1)) < mask_radio
        random_mask = random_mask[None, :, None, None].expand_as(eeg_signal)

        # Apply the mask to the EEG signal
        eeg_signal[random_mask] = 0
        return eeg_signal
    
    def mask_continuous_half(self, eeg_signal, mask_ratio=0.4):
        """
        Continuously mask approximately 50% of the num_patch in the EEG signal.
        """
        bs, num_patch, n_vars, patch_len = eeg_signal.shape

        # Calculate the number of patches to mask
        num_mask_patches = int(num_patch * mask_ratio)

        # Create a mask for each batch
        for i in range(bs):
            # Determine the start index for mask in this batch
            start_idx = torch.randint(0, num_patch - num_mask_patches + 1, (1,)).item()
            
            # Apply the mask
            eeg_signal[i, start_idx:start_idx + num_mask_patches, :, :] = 0

        return eeg_signal


    def run_fine_tone_PatchTST_self_mse(self):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        iteration = 0
        shutil.rmtree("./logs_self_mse")
        self.writer = SummaryWriter("./logs_self_mse")

        criterion = nn.MSELoss()
        self.model.load_state_dict(torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1000_patch50_stride50_epochs-pretrain50_mask0.5_model1.pth'))

        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

            # 训练开始
            self.model.train()
            evaluation = []
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                # printProgressBar(batch_i, train_dataloader.__len__())
                X = X.permute(0, 2, 1)
                X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                origin_X = X
                X = self.mask_continuous_half(X, 0.4)

                X = X.to(self.device)
                X = X.to(torch.float)
                origin_X = origin_X.to(self.device)
                origin_X = origin_X.to(torch.float)

                output = self.model(X)
                train_loss = cosine_distance_loss(output, origin_X)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}".format(epoch+1, train_loss.item()))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.permute(0, 2, 1)
                    X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                    origin_X = X
                    X = self.mask_continuous_half(X, 0.4)

                    X = X.to(self.device)
                    X = X.to(torch.float)
                    origin_X = origin_X.to(self.device)
                    origin_X = origin_X.to(torch.float)

                    output = self.model(X)
                    eval_loss = cosine_distance_loss(output, origin_X)

            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            print("epoch: {}, eval_Loss: {}".format(epoch+1, eval_loss.item()))

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)


    def run_PatchTST_self(self, data_aug=False):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        # 1. 预训练模型
        # saved_state_dict = torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1000_patch50_stride50_epochs-pretrain100_mask0.5_model1.pth')
        # saved_state_dict = torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1921_patch50_stride50_epochs-pretrain100_mask0.5_model1_context1921.pth')

        # 2. 微调后模型
        # saved_state_dict = torch.load('model/PatchTST_self_A03_finetone_mse.pth')

        # 加载预训练模型的 backbone
        # backbone_state_dict = {k[len("backbone."):]: v for k, v in saved_state_dict.items() if k.startswith("backbone.")}
        # self.model.backbone.load_state_dict(backbone_state_dict)

        # for name, param in self.model.named_parameters():   # 冻结 PatchTST 的 backbone 模型
        #     if name.startswith('backbone'):
        #         param.requires_grad = False

        iteration = 0
        best_acc = 0
        shutil.rmtree("./logs_run")
        self.writer = SummaryWriter("./logs_run")
        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

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
                    trans_X = trans.permute(0, 2, 1)
                    trans_X, _ = create_patch(trans_X, 50, 50)
                    embed, output = self.model(trans_X)
                    y = y.to(self.device).long()
                    y = torch.cat((y, y), dim=0)
                # 2. original data
                else:
                    X = X.permute(0, 2, 1)
                    X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                    X = X.to(self.device)
                    X = X.to(torch.float)
                    y = y.to(self.device).long()
                    _, output = self.model(X)

                train_loss = self.loss_func(output, y)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            evaluation = [item for sublist in evaluation for item in sublist]
            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar("acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch+1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    X = X.permute(0, 2, 1)
                    X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                    X = X.to(self.device)
                    X = X.to(torch.float)
                    y = y.to(self.device).long()

                    _, output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    evaluation.append((predicted == y).tolist())
                    eval_loss = self.loss_func(output, y)

            evaluation = [item for sublist in evaluation for item in sublist]
            eval_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar("acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch+1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)
        print("best acc: {}".format(best_acc))

        return train_acc, best_acc


    def run_PatchTST_self_contrast(self):
        # optimizer 按照余弦退火的方式调整学习率
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-4)

        # 1. 预训练模型
        saved_state_dict = torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1000_patch50_stride50_epochs-pretrain100_mask0.5_model1.pth')
        # 2. 微调后模型
        # saved_state_dict = torch.load('model/PatchTST_self_A03_finetone_mse.pth')

        backbone_state_dict = {k[len("backbone."):]: v for k, v in saved_state_dict.items() if k.startswith("backbone.")}
        self.model.backbone.load_state_dict(backbone_state_dict)

        # for name, param in self.model.named_parameters():   # 冻结 PatchTST 的 backbone 模型
        #     if name.startswith('backbone'):
        #         param.requires_grad = False

        iteration = 0
        best_acc = 0
        shutil.rmtree("./logs_run")
        self.writer = SummaryWriter("./logs_run")
        for epoch in range(self.epochs):

            print("——————第 {} 轮训练开始——————".format(epoch + 1))

            # 训练开始
            self.model.train()
            evaluation = []
            lamda = 0.6
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                origin_X = X
                X = X.permute(0, 2, 1)
                X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                X = X.to(self.device)
                X = X.to(torch.float)
                y = y.to(self.device).long()

                _, output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())
                label_loss = self.loss_func(output, y)

                trans = []
                for i in range(origin_X.shape[0]):
                    t1 = transform(origin_X[i], 'permute')
                    trans.append(t1)
                for i in range(origin_X.shape[0]):
                    t2 = transform(origin_X[i], 'cutout_resize')
                    trans.append(t2)
                trans = np.reshape(np.concatenate(trans), (-1, origin_X.shape[1], origin_X.shape[2]))
                trans = torch.tensor(trans, dtype=torch.float).to(self.device)
                trans_X = trans.permute(0, 2, 1)
                trans_X, _ = create_patch(trans_X, 50, 50)
                embed, output = self.model(trans_X)
                contrastive_loss = contrast_loss(embed, temperature=self.temperature)

                train_loss = lamda * label_loss + (1 - lamda) * contrastive_loss

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                iteration += 1

            evaluation = [item for sublist in evaluation for item in sublist]
            train_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/train_loss", train_loss.item(), iteration)
            self.writer.add_scalar("acc/train_acc", train_acc, iteration)
            scheduler.step()
            print("epoch: {}, train_Loss: {}, train_acc: {}".format(epoch+1, train_loss.item(), train_acc))

            # 测试步骤开始
            self.model.eval()
            evaluation = []
            with torch.no_grad():
                for batch_i, (X, y) in enumerate(self.test_dataloader):
                    origin_X = X
                    X = X.permute(0, 2, 1)
                    X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                    X = X.to(self.device)
                    X = X.to(torch.float)
                    y = y.to(self.device).long()

                    _, output = self.model(X)
                    _, predicted = torch.max(output.data, 1)
                    evaluation.append((predicted == y).tolist())
                    label_loss = self.loss_func(output, y)

                    trans = []
                    for i in range(origin_X.shape[0]):
                        t1 = transform(origin_X[i], 'permute')
                        trans.append(t1)
                    for i in range(origin_X.shape[0]):
                        t2 = transform(origin_X[i], 'cutout_resize')
                        trans.append(t2)
                    trans = np.reshape(np.concatenate(trans), (-1, origin_X.shape[1], origin_X.shape[2]))
                    trans = torch.tensor(trans, dtype=torch.float).to(self.device)
                    trans_X = trans.permute(0, 2, 1)
                    trans_X, _ = create_patch(trans_X, 50, 50)
                    embed, output = self.model(trans_X)
                    contrastive_loss = contrast_loss(embed, temperature=self.temperature)

                    eval_loss = lamda * label_loss + (1 - lamda) * contrastive_loss

            evaluation = [item for sublist in evaluation for item in sublist]
            eval_acc = sum(evaluation) / len(evaluation)
            self.writer.add_scalar("loss/eval_loss", eval_loss.item(), iteration)
            self.writer.add_scalar("acc/eval_acc", eval_acc, iteration)
            print("epoch: {}, eval_Loss: {}, eval_acc: {}".format(epoch+1, eval_loss.item(), eval_acc))

            if eval_acc > best_acc:
                best_acc = eval_acc

        self.writer.close()
        torch.save(self.model.state_dict(), self.model_save_path)
        print("best acc: {}".format(best_acc))


    def test_PatchTST_self(self):
        saved_state_dict = torch.load('model/PatchTST_self_A01-A07_classification_finetone.pth')
        self.model.load_state_dict(saved_state_dict)

        iteration = 0
        best_acc = 0

        self.model.eval()
        evaluation = []
        with torch.no_grad():
            for batch_i, (X, y) in enumerate(self.train_dataloader):
                X = X.permute(0, 2, 1)
                X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                X = X.to(self.device)
                X = X.to(torch.float)
                y = y.to(self.device).long()

                _, output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())

            for batch_i, (X, y) in enumerate(self.test_dataloader):
                X = X.permute(0, 2, 1)
                X, _ = create_patch(X, 50, 50)     # [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
                X = X.to(self.device)
                X = X.to(torch.float)
                y = y.to(self.device).long()

                _, output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                evaluation.append((predicted == y).tolist())

        evaluation = [item for sublist in evaluation for item in sublist]
        eval_acc = sum(evaluation) / len(evaluation)
        print("eval_acc: {}".format(eval_acc))
        