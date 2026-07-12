import numpy as np
from .tensor import no_grad


class Optimizer:
    def __init__(self, parameters, lr=0.001):
        self.parameters = list(parameters)
        self.lr = lr
        self._step_count = 0

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        self._step_count += 1

    def state_dict(self):
        return {"lr": self.lr, "step": self._step_count}

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self._step_count = state_dict.get("step", 0)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._velocities = [np.zeros_like(p.data) if momentum > 0 else None for p in self.parameters]

    def step(self):
        super().step()
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                # Gradients are discarded by zero_grad after the update; do
                # not clone an entire parameter tensor just to read them.
                grad = p.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * p.data
                if self.momentum > 0:
                    self._velocities[i] *= self.momentum
                    self._velocities[i] -= self.lr * grad
                    if self.nesterov:
                        p.data += self.momentum * self._velocities[i] - self.lr * grad
                    else:
                        p.data += self._velocities[i]
                else:
                    p.data -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        super().step()
        b1, b2 = self.betas
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * p.data
                m, v = self._m[i], self._v[i]
                m *= b1
                m += (1 - b1) * grad
                v *= b2
                v += (1 - b2) * (grad * grad)
                # Reuse the denominator buffer instead of retaining m_hat,
                # v_hat and sqrt(v_hat) at once for every large parameter.
                denom = np.sqrt(v)
                denom /= np.sqrt(1 - b2 ** self._step_count)
                denom += self.eps
                np.divide(m, denom, out=denom)
                p.data -= (self.lr / (1 - b1 ** self._step_count)) * denom

    def state_dict(self):
        state = super().state_dict()
        state["betas"] = self.betas
        state["eps"] = self.eps
        state["weight_decay"] = self.weight_decay
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.betas = state_dict.get("betas", self.betas)
        self.eps = state_dict.get("eps", self.eps)
        self.weight_decay = state_dict.get("weight_decay", self.weight_decay)


class AdamW(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        super().step()
        b1, b2 = self.betas
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay > 0:
                    p.data *= 1 - self.lr * self.weight_decay
                m, v = self._m[i], self._v[i]
                m *= b1
                m += (1 - b1) * grad
                v *= b2
                v += (1 - b2) * (grad * grad)
                denom = np.sqrt(v)
                denom /= np.sqrt(1 - b2 ** self._step_count)
                denom += self.eps
                np.divide(m, denom, out=denom)
                p.data -= (self.lr / (1 - b1 ** self._step_count)) * denom

    def state_dict(self):
        state = super().state_dict()
        state["betas"] = self.betas
        state["eps"] = self.eps
        state["weight_decay"] = self.weight_decay
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.betas = state_dict.get("betas", self.betas)
        self.eps = state_dict.get("eps", self.eps)
        self.weight_decay = state_dict.get("weight_decay", self.weight_decay)


class RMSprop(Optimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self._sq = [np.zeros_like(p.data) for p in self.parameters]
        self._buf = [np.zeros_like(p.data) if momentum > 0 else None for p in self.parameters]

    def step(self):
        super().step()
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * p.data
                self._sq[i] *= self.alpha
                self._sq[i] += (1 - self.alpha) * (grad * grad)
                step = self.lr * grad / (np.sqrt(self._sq[i]) + self.eps)
                if self.momentum > 0:
                    self._buf[i] = self.momentum * self._buf[i] + step
                    p.data -= self._buf[i]
                else:
                    p.data -= step


class LRScheduler:
    def __init__(self, optimizer, factor=0.1, patience=5, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.optimizer.lr
                self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.wait = 0
                if self.verbose:
                    print(f"  LR reduced: {old_lr:.6f} -> {self.optimizer.lr:.6f}")
