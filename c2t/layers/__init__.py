import numpy as np
from ..tensor import Tensor, no_grad, is_grad_enabled, _Function
from ..accelerator import matmul
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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


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

        if self.training and is_grad_enabled():
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
        self.conv = conv
        self.cols = cols
        self.N = N
        self.OH = OH
        self.OW = OW
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self.backward

    def backward(self, grad_output):
        return self._backward_impl(self.inputs, self.conv, self.cols, self.N, self.OH, self.OW, grad_output)

    @staticmethod
    def _backward_impl(inputs, conv, cols, N, OH, OW, grad_output):
        N, C, H, W = inputs[0].shape
        KH, KW = conv.kernel_size
        SH, SW = conv.stride
        PH, PW = conv.padding

        grad_out_reshaped = np.ascontiguousarray(
            grad_output.transpose(0, 2, 3, 1).reshape(N * OH * OW, -1)
        )

        grad_x = grad_w = None
        if inputs[0].requires_grad:
            w_cols = conv.weight.data.reshape(conv.out_channels, -1)
            grad_cols = grad_out_reshaped @ w_cols
            grad_cols_reshaped = grad_cols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
            grad_x = np.zeros((N, C, H + 2 * PH, W + 2 * PW), dtype=np.float32)
            for i in range(KH):
                for j in range(KW):
                    grad_x[:, :, i:i + OH * SH:SH, j:j + OW * SW:SW] += grad_cols_reshaped[:, :, i, j, :, :]
            if PH > 0:
                grad_x = grad_x[:, :, PH:-PH, :]
            if PW > 0:
                grad_x = grad_x[:, :, :, PW:-PW]
        if inputs[1].requires_grad:
            grad_w_cols = cols.T @ grad_out_reshaped
            grad_w = grad_w_cols.T.reshape(conv.weight.shape)

        grads = [grad_x, grad_w]
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
        mask_tensor = Tensor(mask)
        return x._op(FnDropout, mask_tensor)


class FnBatchNorm(_Function):
    @staticmethod
    def forward_raw(x, gamma, beta, eps, running_mean, running_var, momentum, training, axes, ndim):
        if training:
            mean = x.data.mean(axis=axes)
            var = x.data.var(axis=axes) + eps
            if ndim == 4:
                mean_r = mean.reshape(1, -1, 1, 1)
                var_r = var.reshape(1, -1, 1, 1)
            else:
                mean_r = mean
                var_r = var
            with no_grad():
                running_mean.data = momentum * running_mean.data + (1 - momentum) * mean
                running_var.data = momentum * running_var.data + (1 - momentum) * var
            x_norm = (x.data - mean_r) / np.sqrt(var_r)
        else:
            if ndim == 4:
                rm = running_mean.data.reshape(1, -1, 1, 1)
                rv = running_var.data.reshape(1, -1, 1, 1)
            else:
                rm = running_mean.data
                rv = running_var.data
            x_norm = (x.data - rm) / np.sqrt(rv)
        if ndim == 4:
            gamma_r = gamma.data.reshape(1, -1, 1, 1)
            beta_r = beta.data.reshape(1, -1, 1, 1)
        else:
            gamma_r = gamma.data
            beta_r = beta.data
        return gamma_r * x_norm + beta_r

    @staticmethod
    def forward(ctx, x, gamma, beta, eps, running_mean, running_var, momentum, training, axes, ndim):
        if training:
            mean = x.data.mean(axis=axes)
            var = x.data.var(axis=axes) + eps
            if ndim == 4:
                mean_r = mean.reshape(1, -1, 1, 1)
                var_r = var.reshape(1, -1, 1, 1)
            else:
                mean_r = mean
                var_r = var
            # x and gamma are already retained in ctx.inputs.  Storing copies
            # here used one extra activation-sized buffer per BatchNorm.  The
            # running statistics were already updated by forward_raw; updating
            # them here as well made each training forward count twice.
            ctx.save_for_backward = (mean, var, axes, ndim)
        else:
            if ndim == 4:
                rm = running_mean.data.reshape(1, -1, 1, 1)
                rv = running_var.data.reshape(1, -1, 1, 1)
            else:
                rm = running_mean.data
                rv = running_var.data
            ctx.save_for_backward = None

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.save_for_backward
        if saved is None:
            return (None, None, None, None, None, None, None, None, None, None)
        mean, var, axes, ndim = saved
        x_data = ctx.inputs[0].data
        gamma_data = ctx.inputs[1].data
        N = x_data.size // mean.size
        if ndim == 4:
            mean_r = mean.reshape(1, -1, 1, 1)
            var_r = var.reshape(1, -1, 1, 1)
            gamma_r = gamma_data.reshape(1, -1, 1, 1)
        else:
            mean_r = mean
            var_r = var
            gamma_r = gamma_data
        x_norm = (x_data - mean_r) / np.sqrt(var_r)
        dx_norm = grad_output * gamma_r
        if ndim == 4:
            dgamma = (grad_output * x_norm).sum(axis=(0, 2, 3))
            dbeta = grad_output.sum(axis=(0, 2, 3))
        else:
            dgamma = (grad_output * x_norm).sum(axis=0)
            dbeta = grad_output.sum(axis=0)
        dx = (1.0 / N) * (1.0 / np.sqrt(var_r)) * (
            N * dx_norm
            - dx_norm.sum(axis=axes, keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=axes, keepdims=True)
        )
        return (dx, dgamma, dbeta, None, None, None, None, None, None, None)


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
        if x.ndim == 4:
            axes = (0, 2, 3)
        else:
            axes = 0
        return x._op(FnBatchNorm, self.gamma, self.beta, self.eps,
                     self.running_mean, self.running_var, self.momentum,
                     self.training, axes, x.ndim)


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


class FnDropout(_Function):
    @staticmethod
    def forward_raw(t, mask):
        return t.data * mask.data

    @staticmethod
    def forward(ctx, t, mask):
        # The mask is an input tensor retained by FunctionCtx.
        pass

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.inputs[1].data
        return (grad_output * mask,)


class FnReLU(_Function):
    @staticmethod
    def forward_raw(t):
        return np.maximum(t.data, 0)

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = ((t.data > 0),)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.save_for_backward
        return (grad_output * mask.astype(np.float32),)


class FnLeakyReLU(_Function):
    @staticmethod
    def forward_raw(t, alpha=0.01):
        return np.where(t.data > 0, t.data, alpha * t.data)

    @staticmethod
    def forward(ctx, t, alpha=0.01):
        ctx.alpha = alpha
        ctx.save_for_backward = ((t.data > 0),)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.save_for_backward
        alpha = ctx.alpha
        return (grad_output * np.where(mask, 1.0, alpha).astype(np.float32),)


class FnSigmoid(_Function):
    @staticmethod
    def forward_raw(t):
        return 1.0 / (1.0 + np.exp(-t.data))

    @staticmethod
    def forward(ctx, t):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        s = ctx.output
        return (grad_output * s * (1 - s),)


class FnTanh(_Function):
    @staticmethod
    def forward_raw(t):
        return np.tanh(t.data)

    @staticmethod
    def forward(ctx, t):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        t_val = ctx.output
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

    @staticmethod
    def backward(ctx, grad_output):
        s = ctx.output
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
        out_data = matmul(x.data, self.weight.data)
        if self.use_bias:
            out_data = out_data + self.bias.data
        np.maximum(out_data, 0, out=out_data)
        return _dense_activation_output(x, self, out_data, "relu")


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
        out_data = matmul(x.data, self.weight.data)
        if self.use_bias:
            out_data = out_data + self.bias.data
        # Keep the temporary to one buffer and reuse the final output.
        out_data = 1.0 / (1.0 + np.exp(-out_data))
        return _dense_activation_output(x, self, out_data, "sigmoid")


def _dense_activation_output(x, layer, out_data, activation):
    """Build a fused dense+activation node without a dense intermediate."""
    inputs = [x, layer.weight]
    if layer.use_bias:
        inputs.append(layer.bias)
    requires_grad = is_grad_enabled() and any(inp.requires_grad for inp in inputs)
    out = Tensor(out_data, requires_grad=requires_grad)
    if requires_grad:
        out._ctx = _DenseActivationCtx(inputs, out_data, activation)
    return out


class _DenseActivationCtx:
    def __init__(self, inputs, output, activation):
        self.inputs = inputs
        self.output = output
        self.activation = activation
        self.needs_input_grad = any(inp.requires_grad for inp in inputs)
        self.backward_fn = self.backward

    def backward(self, grad_output):
        x, weight = self.inputs[:2]
        if self.activation == "relu":
            grad_output = grad_output * (self.output > 0)
        else:
            grad_output = grad_output * self.output * (1.0 - self.output)

        grads = [None, None]
        if x.requires_grad:
            grads[0] = matmul(grad_output, weight.data.T)
        if weight.requires_grad:
            grads[1] = matmul(
                x.data.reshape(-1, x.shape[-1]).T,
                grad_output.reshape(-1, grad_output.shape[-1]),
            )
        if len(self.inputs) == 3:
            grads.append(grad_output.sum(axis=tuple(range(grad_output.ndim - 1)))
                         if self.inputs[2].requires_grad else None)
        return tuple(grads)


#
# Pooling layers
#

class MaxPool2D(Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}"

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        xd = x.data
        if PH > 0 or PW > 0:
            xd = np.pad(xd, ((0,0),(0,0),(PH,PH),(PW,PW)), mode='constant')

        OH = (H + 2*PH - KH)//SH + 1
        OW = (W + 2*PW - KW)//SW + 1

        windows = np.lib.stride_tricks.sliding_window_view(xd, (KH, KW), axis=(-2,-1))
        windows = windows[:, :, ::SH, ::SW, :, :]
        out_data = windows.max(axis=(-2, -1))

        if self.training and is_grad_enabled():
            out = Tensor(out_data, requires_grad=True)
            out._ctx = _MaxPoolCtx(x, self, xd, windows, KH, KW, SH, SW, PH, PW, out_data)
            return out
        return Tensor(out_data)


class _MaxPoolCtx:
    def __init__(self, x, pool, xd, windows, KH, KW, SH, SW, PH, PW, output):
        self.inputs = [x]
        self.pool = pool
        self.xd = xd
        self.windows = windows
        self.KH, self.KW = KH, KW
        self.SH, self.SW = SH, SW
        self.PH, self.PW = PH, PW
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self.backward

    def backward(self, grad_output):
        return self._backward(grad_output)

    def _backward(self, grad_output):
        N, C, H, W = self.inputs[0].shape
        PH, PW = self.PH, self.PW
        SH, SW = self.SH, self.SW
        windows = self.windows
        KH, KW = self.KH, self.KW
        OH = grad_output.shape[2]
        OW = grad_output.shape[3]

        grad_x = np.zeros((N, C, H + 2*PH, W + 2*PW), dtype=np.float32)
        max_vals = windows.max(axis=(-2,-1), keepdims=True)
        mask = (windows == max_vals).astype(np.float32)
        counts = mask.sum(axis=(-2,-1), keepdims=True)
        mask = np.divide(mask, np.maximum(counts, 1), out=np.zeros_like(mask), where=counts>0)

        go_expanded = grad_output[:, :, :, :, None, None]
        grad_windows = go_expanded * mask

        for i in range(KH):
            for j in range(KW):
                grad_x[:, :, i:i+OH*SH:SH, j:j+OW*SW:SW] += grad_windows[:, :, :, :, i, j]

        if PH > 0: grad_x = grad_x[:, :, PH:-PH, :]
        if PW > 0: grad_x = grad_x[:, :, :, PW:-PW]
        return (grad_x,)


class AvgPool2D(Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}"

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        xd = x.data
        if PH > 0 or PW > 0:
            xd = np.pad(xd, ((0,0),(0,0),(PH,PH),(PW,PW)), mode='constant')

        OH = (H + 2*PH - KH)//SH + 1
        OW = (W + 2*PW - KW)//SW + 1

        windows = np.lib.stride_tricks.sliding_window_view(xd, (KH, KW), axis=(-2,-1))
        windows = windows[:, :, ::SH, ::SW, :, :]
        out_data = windows.mean(axis=(-2, -1))

        if self.training and is_grad_enabled():
            out = Tensor(out_data, requires_grad=True)
            out._ctx = _AvgPoolCtx(x, self, xd, KH, KW, SH, SW, PH, PW, out_data)
            return out
        return Tensor(out_data)


class _AvgPoolCtx:
    def __init__(self, x, pool, xd, KH, KW, SH, SW, PH, PW, output):
        self.inputs = [x]
        self.xd = xd
        self.KH, self.KW = KH, KW
        self.SH, self.SW = SH, SW
        self.PH, self.PW = PH, PW
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self.backward
        self.pool = pool

    def backward(self, grad_output):
        return self._backward(grad_output)

    def _backward(self, grad_output):
        N, C, H, W = self.inputs[0].shape
        PH, PW = self.PH, self.PW
        SH, SW = self.SH, self.SW
        KH, KW = self.KH, self.KW
        OH, OW = grad_output.shape[2], grad_output.shape[3]

        grad_x = np.zeros((N, C, H + 2*PH, W + 2*PW), dtype=np.float32)
        scale = 1.0 / (KH * KW)
        go_expanded = grad_output[:, :, :, :, None, None] * scale

        for i in range(KH):
            for j in range(KW):
                grad_x[:, :, i:i+OH*SH:SH, j:j+OW*SW:SW] += go_expanded

        if PH > 0: grad_x = grad_x[:, :, PH:-PH, :]
        if PW > 0: grad_x = grad_x[:, :, :, PW:-PW]
        return (grad_x,)


#
# Embedding layer
#

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        w_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.01
        self.weight = Tensor(w_data, requires_grad=True)

    def extra_repr(self):
        return f"{self.num_embeddings}x{self.embedding_dim}"

    def forward(self, x):
        idx = x.data.astype(np.int64) if hasattr(x, 'data') else x.astype(np.int64)
        idx = np.clip(idx, 0, self.num_embeddings - 1)
        out_data = self.weight.data[idx]

        if self.training and is_grad_enabled():
            out = Tensor(out_data, requires_grad=True)
            out._ctx = _EmbeddingCtx(self, idx, out_data)
            return out
        return Tensor(out_data)


class _EmbeddingCtx:
    def __init__(self, emb, idx, output):
        self.inputs = [emb.weight]
        self.idx = idx
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self.backward

    def backward(self, grad_output):
        return self._backward(grad_output)

    def _backward(self, grad_output):
        grad_w = np.zeros_like(self.inputs[0].data)
        idx_flat = self.idx.ravel()
        go_flat = grad_output.reshape(-1, grad_output.shape[-1])
        np.add.at(grad_w, idx_flat, go_flat)
        return (grad_w,)


#
# LayerNorm
#

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Tensor(np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(self.normalized_shape, dtype=np.float32), requires_grad=True)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}"

    def forward(self, x):
        shape = self.normalized_shape
        reduce_axes = tuple(range(-len(shape), 0))

        mean = x.data.mean(axis=reduce_axes, keepdims=True)
        var = x.data.var(axis=reduce_axes, keepdims=True) + self.eps
        x_norm = (x.data - mean) / np.sqrt(var)

        if self.training and is_grad_enabled():
            out = Tensor(x_norm, requires_grad=True)
            out._ctx = _LayerNormCtx(x, self, x_norm, mean, var, reduce_axes, x_norm)
            if self.elementwise_affine:
                out = out * self.weight + self.bias
            return out

        out = Tensor(x_norm)
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class _LayerNormCtx:
    def __init__(self, x, ln, x_norm, mean, var, reduce_axes, output):
        self.inputs = [x]
        if ln.elementwise_affine:
            self.inputs.append(ln.weight)
            self.inputs.append(ln.bias)
        self.ln = ln
        self.x_norm = x_norm
        self.mean = mean
        self.var = var
        self.reduce_axes = reduce_axes
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self.backward

    def backward(self, grad_output):
        return self._backward(grad_output)

    def _backward(self, grad_output):
        ln = self.ln
        x = self.inputs[0]
        x_norm = self.x_norm
        mean = self.mean
        var = self.var
        reduce_axes = self.reduce_axes
        N = x.data.size / x_norm.size

        if ln.elementwise_affine and len(self.inputs) >= 3:
            grad_weight = (grad_output * x_norm).sum(axis=tuple(range(grad_output.ndim - len(ln.normalized_shape))))
            grad_bias = grad_output.sum(axis=tuple(range(grad_output.ndim - len(ln.normalized_shape))))
            grads = [None, grad_weight, grad_bias]
        else:
            grads = [None]

        dx = (1.0 / N) * (1.0 / np.sqrt(var)) * (
            N * grad_output
            - grad_output.sum(axis=reduce_axes, keepdims=True)
            - x_norm * (grad_output * x_norm).sum(axis=reduce_axes, keepdims=True)
        )
        grads[0] = dx
        return tuple(grads)


#
# Fused Conv2DReLU
#

class Conv2DReLU(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.use_bias = use_bias
        k = 1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        w_data = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, in_channels, *self.kernel_size)).astype(np.float32)
        self.weight = Tensor(w_data, requires_grad=True)
        if use_bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)

    def extra_repr(self):
        return f"in={self.in_channels}, out={self.out_channels}, kernel={self.kernel_size}"

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        xd = x.data
        if PH > 0 or PW > 0:
            xd = np.pad(xd, ((0,0),(0,0),(PH,PH),(PW,PW)), mode='constant')

        OH = (H + 2*PH - KH)//SH + 1
        OW = (W + 2*PW - KW)//SW + 1

        windows = np.lib.stride_tricks.sliding_window_view(xd, (KH, KW), axis=(-2,-1))[:,:,::SH,::SW,:,:]
        windows = np.ascontiguousarray(windows)
        cols = windows.transpose(0,2,3,1,4,5).reshape(N*OH*OW, C*KH*KW)
        w_cols = self.weight.data.reshape(self.out_channels, -1)
        out_data = (cols @ w_cols.T).reshape(N, OH, OW, self.out_channels).transpose(0,3,1,2)
        out_data = np.maximum(out_data, 0)

        if self.training and is_grad_enabled():
            out = Tensor(out_data, requires_grad=True)
            out._ctx = _Conv2DReLUCtx(x, self, cols, N, OH, OW, out_data)
            if self.use_bias:
                out = out + self.bias.reshape(1,-1,1,1)
            return out
        out = Tensor(out_data)
        if self.use_bias:
            out = out + self.bias.reshape(1,-1,1,1)
        return out


class _Conv2DReLUCtx:
    def __init__(self, x, conv, cols, N, OH, OW, output):
        self.inputs = [x, conv.weight]
        self.conv = conv
        self.cols = cols
        self.N = N
        self.OH = OH
        self.OW = OW
        self.output = output
        self.needs_input_grad = True
        self.backward_fn = self.backward

    def backward(self, grad_output):
        grad = grad_output * (self.output > 0)
        return _Conv2DCtx._backward_impl(self.inputs, self.conv, self.cols, self.N, self.OH, self.OW, grad)


#
# MultiheadAttention
#

class MultiheadAttention(Module):
    def __init__(self, embed_dim, nhead, dropout=0.0, use_bias=True):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.dropout = dropout

        self.q_proj = Dense(embed_dim, embed_dim, use_bias=use_bias)
        self.k_proj = Dense(embed_dim, embed_dim, use_bias=use_bias)
        self.v_proj = Dense(embed_dim, embed_dim, use_bias=use_bias)
        self.out_proj = Dense(embed_dim, embed_dim, use_bias=use_bias)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, nhead={self.nhead}"

    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = query
        if value is None:
            value = key

        N, Tq, D = query.shape
        _, Tk, _ = key.shape
        _, Tv, _ = value.shape
        H = self.nhead
        hd = self.head_dim

        q = self.q_proj(query).reshape(N, Tq, H, hd).transpose(axes=(0, 2, 1, 3))
        k = self.k_proj(key).reshape(N, Tk, H, hd).transpose(axes=(0, 2, 1, 3))
        v = self.v_proj(value).reshape(N, Tv, H, hd).transpose(axes=(0, 2, 1, 3))

        scale = 1.0 / np.sqrt(hd)
        attn_scores = (q @ k.transpose(axes=(0, 1, 3, 2))) * scale

        attn = attn_scores._op(FnSoftmax, axis=-1)
        out = attn @ v

        out = out.transpose(axes=(0, 2, 1, 3)).reshape(N, Tq, D)
        return self.out_proj(out)


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Dense(d_model, dim_feedforward)
        self.linear2 = Dense(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
        self.activation = ReLU() if activation == "relu" else activation

    def extra_repr(self):
        return f"d_model={self.linear1.in_features}, nhead={self.self_attn.nhead}"

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        if self.training and self.dropout > 0:
            attn_out = Dropout(self.dropout)(attn_out)
        x = self.norm1(x + attn_out)

        ff_out = self.linear1(x)
        ff_out = self.activation(ff_out)
        if self.training and self.dropout > 0:
            ff_out = Dropout(self.dropout)(ff_out)
        ff_out = self.linear2(ff_out)
        if self.training and self.dropout > 0:
            ff_out = Dropout(self.dropout)(ff_out)
        x = self.norm2(x + ff_out)
        return x


class Residual(Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x, *args, **kwargs):
        return x + self.sublayer(x, *args, **kwargs)
