import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple


class Precision_loss(nn.Module):

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        precision = tp / ((tp + fp + self.epsilon))

        precision = precision.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - precision[0].mean()


class F1_Loss(nn.Module):

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        num_p = y_true[:, 1].sum().to(torch.float32)
        num_n = y_true[:, 0].sum().to(torch.float32)
        nums = y_true.sum(dim=0).to(torch.float32)

        alpha_p = num_n/(num_n+num_p)
        alpha_n = num_p/(num_n+num_p)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        ntp = ((1-y_true) * (1-y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        nfn = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)

        n_recall = (ntp / (ntp + nfn + self.epsilon))  # *alpha_n
        precision = (tp / (tp + fp + self.epsilon))  # *alpha_p
        f1 = 2 * (precision*n_recall) / (precision + n_recall + self.epsilon)

        # f1 = precision#[0,:]
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
