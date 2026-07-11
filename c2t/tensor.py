import numpy as np
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, Callable

_GRAD_ENABLED = True

@contextmanager
def no_grad():
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


class Tensor:
    __slots__ = ('data', 'grad', 'requires_grad', '_ctx')

    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=dtype or np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __getitem__(self, key):
        return self._op(FnGetItem, key)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[key] = value

    def numpy(self):
        return self.data.copy()

    def item(self):
        return self.data.item()

    def copy(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def zero_grad(self):
        self.grad = None

    def _op(self, fn_cls, *args, requires_grad=None, **kwargs):
        if not _GRAD_ENABLED:
            ctx = None
            out_data = fn_cls._forward_raw(self, *args, **kwargs)
        elif requires_grad is not None:
            if requires_grad:
                ctx = fn_cls._forward(self, *args, **kwargs)
                out_data = ctx.output
            else:
                ctx = None
                out_data = fn_cls._forward_raw(self, *args, **kwargs)
        else:
            rg = self.requires_grad
            if not rg:
                for arg in args:
                    if isinstance(arg, Tensor) and arg.requires_grad:
                        rg = True
                        break
            if rg:
                ctx = fn_cls._forward(self, *args, **kwargs)
                out_data = ctx.output
            else:
                ctx = None
                out_data = fn_cls._forward_raw(self, *args, **kwargs)
        out = object.__new__(Tensor)
        out.data = np.asarray(out_data, dtype=np.float32)
        out.grad = None
        out.requires_grad = rg if _GRAD_ENABLED else False
        out._ctx = ctx
        return out

    def backward(self, gradient=None):
        if gradient is None:
            assert self.data.ndim == 0, "gradient must be specified for non-scalar tensors"
            gradient = np.float32(1.0)
        elif isinstance(gradient, Tensor):
            gradient = gradient.data

        if self._ctx is None:
            self.grad = np.asarray(gradient, dtype=np.float32)
            return

        topo = []
        visited = set()
        stack = [(self, 0)]
        while stack:
            node, state = stack.pop()
            if state:
                topo.append(node)
                continue
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)
            ctx = node._ctx
            if ctx is None or not ctx.needs_input_grad:
                continue
            stack.append((node, 1))
            for inp in ctx.inputs:
                if isinstance(inp, Tensor) and inp.requires_grad and id(inp) not in visited:
                    stack.append((inp, 0))

        self.grad = np.asarray(gradient, dtype=np.float32)
        for node in reversed(topo):
            grad = node.grad
            if grad is None:
                continue
            ctx = node._ctx
            if ctx is None:
                continue
            grads = ctx.backward(grad)
            for inp, g in zip(ctx.inputs, grads):
                if isinstance(inp, Tensor) and inp.requires_grad and g is not None:
                    if inp.grad is None:
                        inp.grad = np.asarray(g, dtype=np.float32)
                    else:
                        inp.grad += g

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self._op(FnAdd, other)

    def __radd__(self, other):
        return self._op(FnAdd, Tensor(other))

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self._op(FnSub, other)

    def __rsub__(self, other):
        return Tensor(other)._op(FnSub, self)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self._op(FnMul, other)

    def __rmul__(self, other):
        return self._op(FnMul, Tensor(other))

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self._op(FnDiv, other)

    def __rtruediv__(self, other):
        return Tensor(other)._op(FnDiv, self)

    def __neg__(self):
        return self._op(FnNeg)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self._op(FnPow, other)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self._op(FnMatMul, other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return self.data > other

    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return self.data < other

    def sum(self, axis=None, keepdims=False):
        return self._op(FnSum, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return self._op(FnMean, axis=axis, keepdims=keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._op(FnReshape, shape=shape)

    def transpose(self, axes=None):
        return self._op(FnTranspose, axes=axes)

    @property
    def T(self):
        return self._op(FnTranspose, axes=None)

    def exp(self):
        return self._op(FnExp)

    def log(self):
        return self._op(FnLog)

    def sqrt(self):
        return self._op(FnSqrt)

    def square(self):
        return self._op(FnSquare)

    def abs(self):
        return self._op(FnAbs)

    def max(self, axis=None, keepdims=False):
        return self._op(FnMax, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return self._op(FnMin, axis=axis, keepdims=keepdims)

    def clip(self, min_val, max_val):
        return self._op(FnClip, min_val=min_val, max_val=max_val)

    def pad(self, pad_width, mode='constant', constant_values=0):
        return self._op(FnPad, pad_width=pad_width, mode=mode, constant_values=constant_values)

    def flatten(self):
        return self._op(FnReshape, shape=(-1,))

    def logsumexp(self, axis=-1, keepdims=True):
        return self._op(FnLogSumExp, axis=axis, keepdims=keepdims)

    def unsqueeze(self, dim):
        return self._op(FnUnsqueeze, dim=dim)

    def squeeze(self, dim=None):
        return self._op(FnSqueeze, dim=dim)

    def __len__(self):
        return len(self.data)

    def __iadd__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        self.data -= other
        return self

    def __imul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        self.data *= other
        return self

    def __idiv__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        self.data /= other
        return self


class FunctionCtx:
    __slots__ = ('inputs', 'output', 'backward_fn', 'save_for_backward', 'needs_input_grad', 'axis', 'keepdims', 'shape', 'n', 'key', 'axes', 'pad_width', 'dim', 'min_val', 'max_val')

    def __init__(self, inputs, output, backward_fn):
        self.inputs = inputs
        self.output = output
        self.backward_fn = backward_fn
        self.save_for_backward = None
        self.needs_input_grad = any(
            isinstance(inp, Tensor) and inp.requires_grad for inp in inputs
        )

    def backward(self, grad_output):
        return self.backward_fn(self, grad_output)


class _FunctionMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        if name != '_Function':
            raw = dct.get('forward_raw')
            if raw:
                setattr(cls, '_forward_raw', staticmethod(raw))
        return cls


class _Function(metaclass=_FunctionMeta):
    @staticmethod
    def forward_raw(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def _forward(cls, *inputs, **kwargs):
        raw_out = cls.forward_raw(*inputs, **kwargs)
        ctx = FunctionCtx(inputs, raw_out, cls.backward)
        cls.forward(ctx, *inputs, **kwargs)
        return ctx


class FnGetItem(_Function):
    @staticmethod
    def forward_raw(t, key):
        return t.data[key]

    @staticmethod
    def forward(ctx, t, key):
        ctx.key = key
        ctx.shape = t.shape

    @staticmethod
    def backward(ctx, grad_output):
        grad = np.zeros(ctx.shape, dtype=np.float32)
        grad[ctx.key] += grad_output
        return (grad,)


def _reduce_grad(grad, in_shape):
    if grad.shape == in_shape:
        return grad
    nd = grad.ndim - len(in_shape)
    if nd > 0:
        grad = grad.sum(axis=tuple(range(nd)))
    for dim, sz in enumerate(in_shape):
        if sz == 1 and grad.shape[dim] != 1:
            grad = grad.sum(axis=dim, keepdims=True)
    return grad.reshape(in_shape)


class FnAdd(_Function):
    forward_raw = staticmethod(lambda a, b: a.data + b.data)

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward = (a.shape, b.shape)

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.save_for_backward
        return (_reduce_grad(grad_output, a_shape), _reduce_grad(grad_output, b_shape))


class FnSub(_Function):
    forward_raw = staticmethod(lambda a, b: a.data - b.data)

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward = (a.shape, b.shape)

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.save_for_backward
        return (_reduce_grad(grad_output, a_shape), -_reduce_grad(grad_output, b_shape))


class FnMul(_Function):
    forward_raw = staticmethod(lambda a, b: a.data * b.data)

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward = (a.data.copy(), b.data.copy(), a.shape, b.shape)

    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data, a_shape, b_shape = ctx.save_for_backward
        return (_reduce_grad(grad_output * b_data, a_shape),
                _reduce_grad(grad_output * a_data, b_shape))


class FnDiv(_Function):
    forward_raw = staticmethod(lambda a, b: a.data / b.data)

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward = (a.data.copy(), b.data.copy(), a.shape, b.shape)

    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data, a_shape, b_shape = ctx.save_for_backward
        return (_reduce_grad(grad_output / b_data, a_shape),
                -_reduce_grad(grad_output * a_data / (b_data ** 2), b_shape))


class FnNeg(_Function):
    forward_raw = staticmethod(lambda t: -t.data)

    @staticmethod
    def forward(ctx, t):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return (-grad_output,)


class FnPow(_Function):
    forward_raw = staticmethod(lambda a, b: a.data ** b.data)

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward = (a.data.copy(), b.data.copy())

    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.save_for_backward
        return (grad_output * b_data * np.power(a_data, b_data - 1),
                grad_output * np.power(a_data, b_data) * np.log(np.maximum(a_data, 1e-38)))


class FnMatMul(_Function):
    forward_raw = staticmethod(lambda a, b: a.data @ b.data)

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward = (a.shape, b.shape, a.data.copy(), b.data.copy())

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape, a_data, b_data = ctx.save_for_backward
        grad_a = grad_output @ np.swapaxes(b_data, -2, -1)
        grad_b = np.swapaxes(a_data, -2, -1) @ grad_output
        while grad_a.ndim > len(a_shape):
            grad_a = grad_a.sum(axis=0)
        while grad_b.ndim > len(b_shape):
            grad_b = grad_b.sum(axis=0)
        return (grad_a, grad_b)


class FnSum(_Function):
    @staticmethod
    def forward_raw(t, axis=None, keepdims=False):
        return t.data.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def forward(ctx, t, axis=None, keepdims=False):
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.shape = t.shape

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        axis = ctx.axis
        keepdims = ctx.keepdims
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad_output = np.expand_dims(grad_output, axis)
            else:
                for ax in sorted(axis):
                    grad_output = np.expand_dims(grad_output, ax)
        return (np.broadcast_to(grad_output, shape).astype(np.float32),)


class FnMean(_Function):
    @staticmethod
    def forward_raw(t, axis=None, keepdims=False):
        return t.data.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def forward(ctx, t, axis=None, keepdims=False):
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.shape = t.shape
        if axis is None:
            ctx.n = t.data.size
        elif isinstance(axis, int):
            ctx.n = t.data.shape[axis]
        else:
            ctx.n = 1
            for ax in axis:
                ctx.n *= t.data.shape[ax]

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        axis = ctx.axis
        keepdims = ctx.keepdims
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad_output = np.expand_dims(grad_output, axis)
            else:
                for ax in sorted(axis):
                    grad_output = np.expand_dims(grad_output, ax)
        return (np.broadcast_to(grad_output, shape).astype(np.float32) / ctx.n,)


class FnReshape(_Function):
    forward_raw = staticmethod(lambda t, shape: t.data.reshape(shape))

    @staticmethod
    def forward(ctx, t, shape):
        ctx.shape = t.shape

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.reshape(ctx.shape),)


class FnTranspose(_Function):
    @staticmethod
    def forward_raw(t, axes=None):
        return np.transpose(t.data, axes=axes)

    @staticmethod
    def forward(ctx, t, axes=None):
        ctx.axes = axes
        ctx.shape = t.shape

    @staticmethod
    def backward(ctx, grad_output):
        axes = ctx.axes
        if axes is None:
            return (np.transpose(grad_output),)
        return (np.transpose(grad_output, axes=np.argsort(axes)),)


class FnExp(_Function):
    forward_raw = staticmethod(lambda t: np.exp(t.data))

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (np.exp(t.data.copy()),)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * ctx.save_for_backward[0],)


class FnLog(_Function):
    forward_raw = staticmethod(lambda t: np.log(t.data))

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (t.data.copy(),)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output / ctx.save_for_backward[0],)


class FnSqrt(_Function):
    forward_raw = staticmethod(lambda t: np.sqrt(t.data))

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (np.sqrt(t.data.copy()),)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output / (2 * ctx.save_for_backward[0]),)


class FnSquare(_Function):
    forward_raw = staticmethod(lambda t: t.data ** 2)

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (t.data.copy(),)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * 2 * ctx.save_for_backward[0],)


class FnAbs(_Function):
    forward_raw = staticmethod(lambda t: np.abs(t.data))

    @staticmethod
    def forward(ctx, t):
        ctx.save_for_backward = (t.data.copy(),)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * np.sign(ctx.save_for_backward[0]),)


class FnClip(_Function):
    @staticmethod
    def forward_raw(t, min_val, max_val):
        return np.clip(t.data, min_val, max_val)

    @staticmethod
    def forward(ctx, t, min_val, max_val):
        ctx.save_for_backward = (t.data.copy(), min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        data, min_val, max_val = ctx.save_for_backward
        mask = (data >= min_val) & (data <= max_val)
        return (grad_output * mask,)


class FnPad(_Function):
    @staticmethod
    def forward_raw(t, pad_width, mode='constant', constant_values=0):
        return np.pad(t.data, pad_width, mode=mode, constant_values=constant_values)

    @staticmethod
    def forward(ctx, t, pad_width, mode='constant', constant_values=0):
        ctx.pad_width = pad_width
        ctx.shape = t.shape

    @staticmethod
    def backward(ctx, grad_output):
        pad_width = ctx.pad_width
        shape = ctx.shape
        slices = tuple(slice(pw[0], pw[0] + s) for s, pw in zip(shape, pad_width[:len(shape)]))
        return (grad_output[slices],)


class FnUnsqueeze(_Function):
    @staticmethod
    def forward_raw(t, dim):
        return np.expand_dims(t.data, dim)

    @staticmethod
    def forward(ctx, t, dim):
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        return (np.squeeze(grad_output, axis=ctx.dim),)


class FnSqueeze(_Function):
    @staticmethod
    def forward_raw(t, dim=None):
        return np.squeeze(t.data, axis=dim)

    @staticmethod
    def forward(ctx, t, dim=None):
        ctx.dim = dim
        ctx.shape = t.shape

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.reshape(ctx.shape),)


class FnLogSumExp(_Function):
    @staticmethod
    def forward_raw(t, axis=-1, keepdims=True):
        max_val = np.max(t.data, axis=axis, keepdims=True)
        return max_val + np.log(np.sum(np.exp(t.data - max_val), axis=axis, keepdims=keepdims))

    @staticmethod
    def forward(ctx, t, axis=-1, keepdims=True):
        ctx.axis = axis
        ctx.keepdims = keepdims
        max_val = np.max(t.data, axis=axis, keepdims=True)
        ctx.save_for_backward = (t.data.copy(), max_val)

    @staticmethod
    def backward(ctx, grad_output):
        data, max_val = ctx.save_for_backward
        axis = ctx.axis
        keepdims = ctx.keepdims
        exp_stable = np.exp(data - max_val)
        sum_exp = np.sum(exp_stable, axis=axis, keepdims=True)
        grad = grad_output * exp_stable / sum_exp
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad = np.expand_dims(grad, axis)
            else:
                for ax in sorted(axis):
                    grad = np.expand_dims(grad, ax)
        return (grad,)


def cat(tensors, axis=0):
    if not tensors:
        raise ValueError("no tensors to concatenate")
    requires_grad = any(t.requires_grad for t in tensors)
    out_data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(out_data, requires_grad=False)
    if requires_grad and _GRAD_ENABLED:
        out.requires_grad = True
        out._ctx = _CatContext(list(tensors), axis, out_data)
    return out


class _CatContext:
    __slots__ = ('inputs', 'output', 'needs_input_grad', 'backward_fn', 'saved_tensors')

    def __init__(self, tensors, axis, output):
        self.inputs = tensors
        self.output = output
        self.backward_fn = self.backward
        self.needs_input_grad = any(t.requires_grad for t in tensors)
        self.saved_tensors = ([t.data.copy() for t in tensors], axis)

    def backward(self, grad_output):
        data_list, axis = self.saved_tensors
        grads = np.split(grad_output, np.cumsum([d.shape[axis] for d in data_list[:-1]]), axis=axis)
        return tuple(grads)


def stack(tensors, axis=0):
    if not tensors:
        raise ValueError("no tensors to stack")
    requires_grad = any(t.requires_grad for t in tensors)
    out_data = np.stack([t.data for t in tensors], axis=axis)
    out = Tensor(out_data, requires_grad=False)
    if requires_grad and _GRAD_ENABLED:
        out.requires_grad = True
        out._ctx = _StackContext(list(tensors), axis, out_data)
    return out


class _StackContext:
    __slots__ = ('inputs', 'output', 'needs_input_grad', 'backward_fn', 'saved_tensors')

    def __init__(self, tensors, axis, output):
        self.inputs = tensors
        self.output = output
        self.backward_fn = self.backward
        self.needs_input_grad = any(t.requires_grad for t in tensors)
        self.saved_tensors = ([t.data.copy() for t in tensors], axis)

    def backward(self, grad_output):
        _, axis = self.saved_tensors
        return tuple(np.squeeze(g, axis=axis) for g in np.split(grad_output, len(self.inputs), axis=axis))
