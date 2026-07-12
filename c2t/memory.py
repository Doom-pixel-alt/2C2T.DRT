import numpy as np
import gc
import os
try:
    import psutil
except ImportError:  # Keep the core dependency-free when psutil is absent.
    psutil = None
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
        if psutil is None:
            return 1024
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


def suggest_batch_size(model, sample_shape, target_mb=None, activation_mb_per_sample=None):
    """Return a conservative batch size based on a measured activation graph.

    ``activation_mb_per_sample`` is accepted for callers that have already
    profiled their model, avoiding a second probe forward pass.
    """
    if target_mb is None:
        target_mb = get_available_memory_mb() * 0.3

    param_mb = estimate_model_size(model)['megabytes']
    # Parameters + gradients + Adam's first and second moments.  This is the
    # common default and deliberately errs on the safe side for auto-batch.
    fixed_mb = param_mb * 4

    if activation_mb_per_sample is None:
        try:
            activation_mb_per_sample = estimate_activation_memory(
                model, sample_shape, batch_size=1
            )['megabytes']
        except (MemoryError, ValueError):
            activation_mb_per_sample = 0.0
    # A model with no differentiable layers still needs at least its input.
    sample_mb = np.zeros((1, *sample_shape), dtype=np.float32).nbytes / (1024 ** 2)
    activations_per_sample = max(float(activation_mb_per_sample), sample_mb)

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


def _split_segments(layers, num_segments):
    num_segments = max(1, min(int(num_segments), len(layers)))
    base, remainder = divmod(len(layers), num_segments)
    segments = []
    offset = 0
    for index in range(num_segments):
        size = base + (1 if index < remainder else 0)
        segments.append(layers[offset:offset + size])
        offset += size
    return segments


def _non_grad_tensor_state(model):
    """Snapshot buffers such as BatchNorm running statistics."""
    return [
        (parameter, parameter.data.copy())
        for parameter in model.parameters()
        if not parameter.requires_grad
    ]


def _restore_tensor_state(state):
    for parameter, data in state:
        parameter.data[...] = data


class GradientCheckpointer:
    """Activation checkpointing for ``Sequential`` training graphs.

    Only the input to each segment is kept.  During backward the segment is
    replayed with its original NumPy RNG state, yielding the exact Dropout mask
    used in forward.  Non-trainable tensor buffers are restored afterwards so
    BatchNorm statistics are updated exactly once per real batch.
    """

    def __init__(self, num_segments=4):
        if num_segments < 1:
            raise ValueError("num_segments must be at least 1")
        self.num_segments = num_segments

    @staticmethod
    def supports(model):
        # Arbitrary Module trees cannot be split without changing their
        # forward semantics (Residual/attention have non-linear call graphs).
        return type(model).__name__ == "Sequential" and bool(model._modules)

    def checkpoint_forward(self, model, x):
        if not self.supports(model):
            raise ValueError(
                "gradient checkpointing currently supports c2t.Sequential models only"
            )

        segments = _split_segments(list(model._modules.values()), self.num_segments)
        boundaries = [x]
        rng_states = []
        with no_grad():
            h = x
            for index, segment in enumerate(segments):
                rng_states.append(np.random.get_state())
                for layer in segment:
                    h = layer(h)
                # The last output is retained by ``output`` below, so keeping
                # it as a boundary would be redundant.
                if index + 1 < len(segments):
                    boundaries.append(h)

        state = {
            "segments": segments,
            "boundaries": boundaries,
            "rng_states": rng_states,
            "post_forward_rng_state": np.random.get_state(),
            "buffer_state": _non_grad_tensor_state(model),
        }
        # Sharing the final array is intentional: the loss needs a leaf from
        # which it can propagate dL/d(output), not a second activation copy.
        return Tensor(h.data, requires_grad=True), state

    def checkpoint_backward(self, loss, output, state):
        loss.backward()
        grad = output.grad
        if grad is None:
            raise RuntimeError("checkpointed output did not receive a gradient")

        try:
            for index in range(len(state["segments"]) - 1, -1, -1):
                np.random.set_state(state["rng_states"][index])
                inp = Tensor(state["boundaries"][index].data, requires_grad=True)
                h = inp
                for layer in state["segments"][index]:
                    h = layer(h)
                h.backward(gradient=grad)
                grad = inp.grad
                if grad is None:
                    raise RuntimeError("checkpoint segment did not produce an input gradient")
                inp.grad = None
        finally:
            np.random.set_state(state["post_forward_rng_state"])
            _restore_tensor_state(state["buffer_state"])
            output.grad = None
            # Release checkpoint boundaries promptly instead of waiting for the
            # next Python garbage-collection cycle.
            state.clear()


def _array_root(array):
    root = array
    while isinstance(getattr(root, 'base', None), np.ndarray):
        root = root.base
    return root


def _collect_graph_arrays(value, arrays, seen_objects, parameter_roots):
    """Collect unique NumPy buffers retained by a live autograd graph."""
    identifier = id(value)
    if identifier in seen_objects:
        return
    seen_objects.add(identifier)

    if isinstance(value, np.ndarray):
        root = _array_root(value)
        if id(root) not in parameter_roots:
            arrays[id(root)] = root.nbytes
        return
    if isinstance(value, Tensor):
        _collect_graph_arrays(value.data, arrays, seen_objects, parameter_roots)
        if value._ctx is not None:
            _collect_graph_arrays(value._ctx, arrays, seen_objects, parameter_roots)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_graph_arrays(item, arrays, seen_objects, parameter_roots)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_graph_arrays(item, arrays, seen_objects, parameter_roots)
        return

    for attribute in getattr(value, '__slots__', ()):  # FunctionCtx
        if hasattr(value, attribute):
            _collect_graph_arrays(getattr(value, attribute), arrays, seen_objects, parameter_roots)
    for item in getattr(value, '__dict__', {}).values():  # Custom layer ctxs
        _collect_graph_arrays(item, arrays, seen_objects, parameter_roots)


def estimate_activation_memory(model, sample_shape, batch_size=1):
    """Measure graph-retained activation bytes for a real probe forward pass.

    Unlike the old input-size heuristic, this accounts for activations saved by
    the model's actual layers.  It restores RNG, training mode and non-gradient
    buffers, so profiling has no effect on subsequent training.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not hasattr(model, 'parameters'):
        raise ValueError("model must expose parameters()")

    module_modes = []
    stack = [model]
    while stack:
        module = stack.pop()
        module_modes.append((module, module.training))
        stack.extend(module._modules.values())
    buffer_state = _non_grad_tensor_state(model)
    rng_state = np.random.get_state()
    output = None
    try:
        model.train()
        x = Tensor(np.zeros((batch_size, *sample_shape), dtype=np.float32))
        output = model(x)
        parameter_roots = {id(_array_root(p.data)) for p in model.parameters()}
        arrays = {}
        _collect_graph_arrays(output, arrays, set(), parameter_roots)
        total = sum(arrays.values())
        return {
            "bytes": total,
            "megabytes": total / (1024 ** 2),
            "per_sample_bytes": total / batch_size,
            "per_sample_megabytes": total / batch_size / (1024 ** 2),
        }
    finally:
        output = None
        _restore_tensor_state(buffer_state)
        np.random.set_state(rng_state)
        for module, training in module_modes:
            module.train(training)
        gc.collect()


def estimate_training_memory(model, sample_shape, batch_size=1, optimizer=None):
    """Estimate the live training footprint from the real autograd graph.

    The result separates persistent model state from batch-dependent
    activations, making the trade-off between micro-batch size and gradient
    accumulation explicit to callers and the CLI.
    """
    params = estimate_model_size(model)['bytes']
    activation = estimate_activation_memory(model, sample_shape, batch_size)
    state_multiplier = 0
    optimizer_name = type(optimizer).__name__ if optimizer is not None else "Adam"
    if optimizer_name in ("Adam", "AdamW"):
        state_multiplier = 2
    elif optimizer_name == "SGD" and getattr(optimizer, 'momentum', 0) > 0:
        state_multiplier = 1
    elif optimizer_name == "RMSprop":
        state_multiplier = 1 + int(getattr(optimizer, 'momentum', 0) > 0)

    gradient_bytes = params
    optimizer_bytes = params * state_multiplier
    total = params + gradient_bytes + optimizer_bytes + activation['bytes']
    return {
        "parameters_bytes": params,
        "gradients_bytes": gradient_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activations_bytes": activation['bytes'],
        "total_bytes": total,
        "parameters_megabytes": params / (1024 ** 2),
        "gradients_megabytes": gradient_bytes / (1024 ** 2),
        "optimizer_megabytes": optimizer_bytes / (1024 ** 2),
        "activations_megabytes": activation['megabytes'],
        "total_megabytes": total / (1024 ** 2),
        "activation_per_sample_megabytes": activation['per_sample_megabytes'],
    }


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
