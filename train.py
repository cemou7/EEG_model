# run_subject_train.py
import logging, os, numpy as np, torch
from torch import nn

from data_processing.process_GCN import SEED3Dataset
from model_set.FCLGCN import GCNTCN
def set_logger(log_path='../logs/train.log'):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
		features: hidden vector of shape [bsz, n_views, ...].形状的隐藏向量[bsz，n_views，…]。
		labels: ground truth of shape [bsz]. 形状的基本真值
		mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
		has the same class as sample i. Can be asymmetric.
		mask:形状对比掩模[bsz，bsz]，掩模{i，j}=1，如果样本j与样本i具有相同的类。i,j可以是不对称的。
		Returns:
		A loss scalar.损失标量
		"""
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        mask = mask * logits_mask  #

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        eps = 1e-30
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

        # loss
        loss = -  mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    

# ---------- 全局配置 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = dict(T=4, batch_size=64, num_epochs=500, lr=3e-4, alpha=8e-4)

model_dir = './model/seed/Alpha_beta'
os.makedirs(f'{model_dir}/seed0',  exist_ok=True)
img_dir   = './imgs/seed/Alpha_beta'
os.makedirs(img_dir, exist_ok=True)

loss_fn         = nn.CrossEntropyLoss().to(device)
contrastiveLoss = SupConLoss(0.1).to(device)

# ---------- 评估 ----------
@torch.no_grad()
def evaluate(model, data_list, label_tensor):
    model.eval()
    correct = 0
    for data, label in zip(data_list, label_tensor):
        data = data.to(device)
        _, pred = model([data])
        correct += (torch.argmax(pred, -1) == label).item()
    return correct / len(label_tensor)

# ---------- 训练+验证 ----------
def train_and_validate(model, train_data, train_labels,
                       test_data,  test_labels,  optimizer):

    best_acc, best_wts = 0., None
    bs = params['batch_size']

    for epoch in range(params['num_epochs']):
        model.train()
        total_loss = 0.
        for i in range(0, len(train_data), bs):
            batch  = [g.to(device) for g in train_data[i:i+bs]]
            labels = train_labels[i:i+bs]

            optimizer.zero_grad()
            proj, pred = model(batch)
            loss = contrastiveLoss(proj, labels)*0.2 + loss_fn(pred, labels)*0.8
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, test_data, test_labels)
        if acc > best_acc:
            best_acc, best_wts = acc, model.state_dict()

        if epoch % 10 == 0:
            logging.info(f'Epoch {epoch:03d}  loss={total_loss:.4f}  val_acc={acc:.4f}')

    model.load_state_dict(best_wts)
    logging.info(f'Finished. best_val_acc={best_acc:.4f}')
    return model, best_acc

# ---------- 主流程 ----------
def subject_dependent_exp():
    logging.info('=== Subject-Dependent Training (train / test split) ===')

    root_dir = '/home/work/CZT/CL-Model/dataset/data_no_reject_TE/'
    subjects = [f'A{i:02d}' for i in range(1, 10)]   # A01~A09

    all_acc = []
    for subj in subjects:
        logging.info(f'\n--- Subject {subj} ---')

        train_set = SEED3Dataset(root=root_dir, subj=subj, mode='train')
        test_set  = SEED3Dataset(root=root_dir, subj=subj, mode='test')

        train_data   = [g for g,_ in train_set]            # Batch 列表
        train_labels = torch.tensor([y for _,y in train_set],
                                    dtype=torch.long, device=device)
        test_data    = [g for g,_ in test_set]
        test_labels  = torch.tensor([y for _,y in test_set],
                                    dtype=torch.long, device=device)

        model = GCNTCN(K=2, T=params['T'],
                       num_channels=22, num_features=5).to(device)
        optim = torch.optim.Adam(model.parameters(),
                                 lr=params['lr'], weight_decay=params['alpha'])

        best_model, best_acc = train_and_validate(
            model, train_data, train_labels, test_data, test_labels, optim)

        save_path = os.path.join(model_dir, 'seed0', f'model_{subj}.pt')
        torch.save(best_model.state_dict(), save_path)
        logging.info(f'Saved best model to {save_path}')
        all_acc.append(best_acc)

    logging.info('=== Finished All Subjects ===')
    logging.info(f'Average test accuracy over {len(subjects)} subjects: {np.mean(all_acc):.4f}')

# ---------- 入口 ----------
if __name__ == '__main__':
    set_logger('./logs/subject_train.log')
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    subject_dependent_exp()
