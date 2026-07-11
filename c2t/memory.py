import numpy as np
import gc
import os
import psutil
from .tensor import Tensor, no_grad


def free_memory():
    gc.collect()
    try:
        import ctypes
        if hasattr(ctypes, 'windll'):
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except:
        pass


def get_available_memory_mb():
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except:
        return 1024


def estimate_model_size(model):
    total_bytes = 0
    total_params = 0
    for p in model.parameters():
        total_bytes += p.data.nbytes
        total_params += p.data.size
    return {
        "parameters": total_params,
        "bytes": total_bytes,
        "kilobytes": total_bytes / 1024,
        "megabytes": total_bytes / (1024 ** 2),
        "gigabytes": total_bytes / (1024 ** 3),
    }


def suggest_batch_size(model, sample_shape, target_mb=None):
    if target_mb is None:
        target_mb = get_available_memory_mb() * 0.3

    model.eval()
    test_x = Tensor(np.random.randn(1, *sample_shape).astype(np.float32))
    try:
        _ = model(test_x)
    except:
        pass

    param_mb = estimate_model_size(model)['megabytes']
    grad_mb = param_mb * 2
    opt_mb = param_mb * 2
    fixed_mb = param_mb + grad_mb + opt_mb

    sample = np.random.randn(1, *sample_shape).astype(np.float32)
    sample_activations = sample.nbytes / (1024 ** 2)
    activations_per_sample = sample_activations * 3

    per_sample_mb = activations_per_sample
    available = max(1, target_mb - fixed_mb)
    batch_size = max(1, int(available / max(per_sample_mb, 0.01)))
    return min(batch_size, 512)


def quantize(tensor_data, bits=8):
    if bits == 8:
        dtype = np.int8
    elif bits == 16:
        dtype = np.int16
    else:
        raise ValueError(f"unsupported bits: {bits}")
    min_val = tensor_data.min()
    max_val = tensor_data.max()
    scale = (max_val - min_val) / (2 ** bits - 1) if max_val > min_val else 1.0
    zero_point = -min_val / scale if scale > 0 else 0
    quantized = np.round(tensor_data / scale + zero_point).astype(dtype)
    return quantized, scale, zero_point


def dequantize(quantized, scale, zero_point):
    return (quantized.astype(np.float32) - zero_point) * scale


class WeightCompressor:
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio

    def compress(self, weight_matrix):
        U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        k = max(1, int(S.size * self.compression_ratio))
        return U[:, :k], S[:k], Vt[:k, :]

    @staticmethod
    def decompress(U, S, Vt):
        return U @ np.diag(S) @ Vt


class GradientCheckpointer:
    def __init__(self, num_checkpoints=4):
        self.num_checkpoints = num_checkpoints
        self._saved_activations = {}

    def checkpoint_forward(self, model, x):
        if not hasattr(model, '_modules'):
            return model(x), []

        layers = list(model._modules.values())
        n = len(layers)
        seg_size = max(1, n // self.num_checkpoints)
        segments = [layers[i:i + seg_size] for i in range(0, n, seg_size)]

        activations = [x.data.copy()]
        h = x
        with no_grad():
            for seg_idx, seg in enumerate(segments):
                for layer in seg:
                    h = layer(h)
                activations.append(h.data.copy())
        out = Tensor(h.data, requires_grad=True)
        return out, activations

    def checkpoint_backward(self, model, loss, output, activations, input_tensor=None):
        if not hasattr(model, '_modules'):
            loss.backward()
            return

        layers = list(model._modules.values())
        n = len(layers)
        seg_size = max(1, n // self.num_checkpoints)
        segments = [layers[i:i + seg_size] for i in range(0, n, seg_size)]

        # Backprop through loss function to get gradient at output
        loss.backward()
        grad = output.grad if output.grad is not None else np.float32(1.0)

        for seg_idx in range(len(segments) - 1, -1, -1):
            if seg_idx == 0 and input_tensor is not None:
                inp = input_tensor
            else:
                inp = Tensor(activations[seg_idx], requires_grad=True)
            h = inp
            for layer in segments[seg_idx]:
                h = layer(h)
            h.backward(gradient=grad)
            if seg_idx > 0:
                grad = inp.grad if inp.grad is not None else np.zeros_like(inp.data)


class MemoryMappedParam:
    def __init__(self, shape, filename=None, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        if filename is None:
            self.filename = os.path.join(
                os.environ.get('TEMP', '/tmp'),
                f'c2t_mmap_{id(self)}.dat'
            )
        else:
            self.filename = filename
        self._mmap = np.memmap(self.filename, dtype=dtype, mode='w+', shape=shape)

    def read(self):
        return np.array(self._mmap)

    def write(self, data):
        self._mmap[:] = data[:]
        self._mmap.flush()

    def as_tensor(self, requires_grad=False):
        t = Tensor(np.array(self._mmap), requires_grad=requires_grad)
        return t

    def close(self):
        del self._mmap

    def __del__(self):
        try:
            del self._mmap
            if os.path.exists(self.filename):
                os.remove(self.filename)
        except:
            pass
