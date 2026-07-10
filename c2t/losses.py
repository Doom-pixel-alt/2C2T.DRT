import numpy as np
from .tensor import Tensor


class Loss:
    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        diff = pred - target
        return (diff ** 2).mean()


class MAELoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        diff = pred - target
        return diff.abs().mean()


class CrossEntropyLoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)

        if target.ndim == 1:
            N = pred.shape[0]
            log_softmax = pred - pred.logsumexp(axis=-1, keepdims=True)
            loss = -log_softmax[range(N), target.data.astype(int)].mean()
        else:
            log_softmax = pred - pred.logsumexp(axis=-1, keepdims=True)
            loss = -(target * log_softmax).sum(axis=-1).mean()

        return loss

    @staticmethod
    def logsumexp(x, axis=-1, keepdims=True):
        if isinstance(x, Tensor):
            x = x.data
        max_val = np.max(x, axis=axis, keepdims=True)
        stable = x - max_val
        return max_val + np.log(np.sum(np.exp(stable), axis=axis, keepdims=keepdims))


class BinaryCrossEntropyLoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        eps = 1e-7
        pred_clip = pred.clip(eps, 1 - eps)
        return -(target * pred_clip.log() + (1 - target) * (1 - pred_clip).log()).mean()


class NLLLoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        N = pred.shape[0]
        return -pred[range(N), target.data.astype(int)].mean()


class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        diff = pred - target
        abs_diff = diff.abs()
        mask = (abs_diff.data <= self.delta).astype(np.float32)
        loss = 0.5 * (diff ** 2) * mask + self.delta * (abs_diff - 0.5 * self.delta) * (1 - mask)
        return loss.mean()
