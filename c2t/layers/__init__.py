import numpy as np
from ..tensor import Tensor, no_grad
from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        if name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def parameters(self):
        params = []
        for p in self._parameters.values():
            params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self):
        named = [(k, v) for k, v in self._parameters.items()]
        for name, module in self._modules.items():
            for sub_name, param in module.named_parameters():
                named.append((f"{name}.{sub_name}", param))
        return named

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)

    def eval(self):
        self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self):
        state = {k: v.data.copy() for k, v in self._parameters.items()}
        for name, module in self._modules.items():
            sub_state = module.state_dict()
            for k, v in sub_state.items():
                state[f"{name}.{k}"] = v
        return state

    def load_state_dict(self, state_dict):
        own_params = dict(self.named_parameters())
        for name, data in state_dict.items():
            if name in own_params:
                own_params[name].data[:] = data

    def extra_repr(self):
        return ""

    def __repr__(self):
        lines = [f"{type(self).__name__}({self.extra_repr()})"]
        for name, module in self._modules.items():
            sub_repr = repr(module)
            for line in sub_repr.split('\n'):
                lines.append(f"  ({name}): {line}")
        return '\n'.join(lines)

    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f"_{i}", layer)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def add(self, layer):
        i = len(self._modules)
        setattr(self, f"_{i}", layer)

    def __getitem__(self, idx):
        keys = list(self._modules.keys())
        return self._modules[keys[idx]]

    def __len__(self):
        return len(self._modules)


class Dense(Module):
    def __init__(self, in_features, out_features, use_bias=True, weight_scale=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        w_data = np.random.randn(in_features, out_features) * weight_scale
        self.weight = Tensor(w_data, requires_grad=True)

        if use_bias:
            b_data = np.zeros(out_features, dtype=np.float32)
            self.bias = Tensor(b_data, requires_grad=True)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.use_bias}"

    def forward(self, x):
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.use_bias = use_bias

        k = 1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        w_data = np.random.uniform(-np.sqrt(k), np.sqrt(k),
                                    size=(out_channels, in_channels, *self.kernel_size)).astype(np.float32)
        self.weight = Tensor(w_data, requires_grad=True)

        if use_bias:
            b_data = np.zeros(out_channels, dtype=np.float32)
            self.bias = Tensor(b_data, requires_grad=True)

    def extra_repr(self):
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, padding={self.padding}")

    def _im2col(self, x_data):
        N, C, H, W = x_data.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        if PH > 0 or PW > 0:
            x_data = np.pad(x_data, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')

        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

        windows = np.lib.stride_tricks.sliding_window_view(
            x_data, (KH, KW), axis=(-2, -1)
        )[:, :, ::SH, ::SW, :, :]

        windows = np.ascontiguousarray(windows)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * OH * OW, C * KH * KW)
        return cols, N, OH, OW

    def forward(self, x):
        N, C, H, W = x.shape
        cols, N, OH, OW = self._im2col(x.data)
        w_cols = self.weight.data.reshape(self.out_channels, -1)
        out_data = (cols @ w_cols.T).reshape(N, OH, OW, self.out_channels).transpose(0, 3, 1, 2)

        if self.training:
            out = Tensor(out_data, requires_grad=True)
            out._ctx = _Conv2DCtx(x, self, cols, N, OH, OW, out_data)
        else:
            out = Tensor(out_data)

        if self.use_bias:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out


class _Conv2DCtx:
    def __init__(self, x, conv, cols, N, OH, OW, output):
        self.inputs = [x, conv.weight]
        if conv.use_bias:
            self.inputs.append(conv.bias)
        self.conv = conv
        self.cols = cols
        self.N = N
        self.OH = OH
        self.OW = OW
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self._backward

    def _backward(self, grad_output):
        conv = self.conv
        N, C, H, W = self.inputs[0].shape
        KH, KW = conv.kernel_size
        SH, SW = conv.stride
        PH, PW = conv.padding

        grad_out_reshaped = np.ascontiguousarray(
            grad_output.transpose(0, 2, 3, 1).reshape(N * self.OH * self.OW, -1)
        )

        w_cols = conv.weight.data.reshape(conv.out_channels, -1)
        grad_cols = grad_out_reshaped @ w_cols
        grad_w_cols = self.cols.T @ grad_out_reshaped
        grad_w = grad_w_cols.T.reshape(conv.weight.shape)

        grad_cols_reshaped = grad_cols.reshape(N, self.OH, self.OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        grad_x = np.zeros((N, C, H + 2 * PH, W + 2 * PW), dtype=np.float32)
        for i in range(KH):
            for j in range(KW):
                grad_x[:, :, i:self.OH * SH:SH, j:self.OW * SW:SW] += grad_cols_reshaped[:, :, i, j, :, :]
        if PH > 0:
            grad_x = grad_x[:, :, PH:-PH, :]
        if PW > 0:
            grad_x = grad_x[:, :, :, PW:-PW]

        grads = [grad_x, grad_w]
        if conv.use_bias:
            grads.append(np.asarray(grad_output.sum(axis=(0, 2, 3))))
        return tuple(grads)


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

    def extra_repr(self):
        return ""


class Dropout(Module):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate

    def extra_repr(self):
        return f"rate={self.rate}"

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        mask = np.random.binomial(1, 1 - self.rate, size=x.shape).astype(np.float32)
        mask /= (1 - self.rate)
        out = Tensor(x.data * mask, requires_grad=x.requires_grad)
        return out


class BatchNorm(Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32))

    def extra_repr(self):
        return f"features={self.num_features}"

    def forward(self, x):
        if self.training:
            if x.ndim == 4:
                axes = (0, 2, 3)
            else:
                axes = 0

            batch_mean = Tensor(x.data.mean(axis=axes))
            batch_var = Tensor(x.data.var(axis=axes) + self.eps)

            with no_grad():
                self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * batch_mean.data
                self.running_var.data = self.momentum * self.running_var.data + (1 - self.momentum) * batch_var.data

            if x.ndim == 4:
                batch_mean_r = batch_mean.reshape(1, -1, 1, 1)
                batch_var_r = batch_var.reshape(1, -1, 1, 1)
                gamma_r = self.gamma.reshape(1, -1, 1, 1)
                beta_r = self.beta.reshape(1, -1, 1, 1)
            else:
                batch_mean_r = batch_mean
                batch_var_r = batch_var
                gamma_r = self.gamma
                beta_r = self.beta

            x_norm = (x - batch_mean_r) / batch_var_r.sqrt()
            return gamma_r * x_norm + beta_r
        else:
            if x.ndim == 4:
                rm = self.running_mean.reshape(1, -1, 1, 1)
                rv = self.running_var.reshape(1, -1, 1, 1)
                g = self.gamma.reshape(1, -1, 1, 1)
                b = self.beta.reshape(1, -1, 1, 1)
            else:
                rm = self.running_mean
                rv = self.running_var
                g = self.gamma
                b = self.beta

            x_norm = (x - rm) / rv.sqrt()
            return g * x_norm + b


class ReLU(Module):
    def extra_repr(self):
        return ""

    def forward(self, x):
        return x._op(FnReLU)


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def extra_repr(self):
        return f"alpha={self.alpha}"

    def forward(self, x):
        return x._op(FnLeakyReLU, alpha=self.alpha)


class Sigmoid(Module):
    def forward(self, x):
        return x._op(FnSigmoid)


class Tanh(Module):
    def forward(self, x):
        return x._op(FnTanh)


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def extra_repr(self):
        return f"axis={self.axis}"

    def forward(self, x):
        return x._op(FnSoftmax, axis=self.axis)


class Identity(Module):
    def forward(self, x):
        return x


class Reshape(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def extra_repr(self):
        return f"shape={self.shape}"

    def forward(self, x):
        return x.reshape(*self.shape)


from ..tensor import _Function

class FnReLU(_Function):
    @staticmethod
    def forward_raw(t):
        return np.maximum(t.data, 0)

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (t.data.copy(),)

    @staticmethod
    def backward(ctx, grad_output):
        (val,) = ctx.save_for_backward
        return (grad_output * (val > 0).astype(np.float32),)


class FnLeakyReLU(_Function):
    @staticmethod
    def forward_raw(t, alpha=0.01):
        return np.where(t.data > 0, t.data, alpha * t.data)

    @staticmethod
    def forward(ctx, t, alpha=0.01):
        ctx.alpha = alpha
        ctx.save_for_backward = (t.data.copy(),)

    @staticmethod
    def backward(ctx, grad_output):
        (val,) = ctx.save_for_backward
        alpha = ctx.alpha
        return (grad_output * np.where(val > 0, 1.0, alpha).astype(np.float32),)


class FnSigmoid(_Function):
    @staticmethod
    def forward_raw(t):
        return 1.0 / (1.0 + np.exp(-t.data))

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (1.0 / (1.0 + np.exp(-t.data.copy())),)

    @staticmethod
    def backward(ctx, grad_output):
        (s,) = ctx.save_for_backward
        return (grad_output * s * (1 - s),)


class FnTanh(_Function):
    @staticmethod
    def forward_raw(t):
        return np.tanh(t.data)

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (np.tanh(t.data.copy()),)

    @staticmethod
    def backward(ctx, grad_output):
        (t_val,) = ctx.save_for_backward
        return (grad_output * (1 - t_val ** 2),)


class FnSoftmax(_Function):
    @staticmethod
    def forward_raw(t, axis=-1):
        max_val = np.max(t.data, axis=axis, keepdims=True)
        e_x = np.exp(t.data - max_val)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def forward(ctx, t, axis=-1):
        ctx.axis = axis
        max_val = np.max(t.data, axis=axis, keepdims=True)
        e_x = np.exp(t.data - max_val)
        ctx.save_for_backward = (e_x / np.sum(e_x, axis=axis, keepdims=True),)

    @staticmethod
    def backward(ctx, grad_output):
        (s,) = ctx.save_for_backward
        axis = ctx.axis
        return (s * (grad_output - (s * grad_output).sum(axis=axis, keepdims=True)),)


class DenseReLU(Module):
    def __init__(self, in_features, out_features, use_bias=True, weight_scale=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w_data = np.random.randn(in_features, out_features) * weight_scale
        self.weight = Tensor(w_data, requires_grad=True)
        if use_bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        self.use_bias = use_bias

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}"

    def forward(self, x):
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out._op(FnReLU)


class DenseSigmoid(Module):
    def __init__(self, in_features, out_features, use_bias=True, weight_scale=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w_data = np.random.randn(in_features, out_features) * weight_scale
        self.weight = Tensor(w_data, requires_grad=True)
        if use_bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        self.use_bias = use_bias

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}"

    def forward(self, x):
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out._op(FnSigmoid)
