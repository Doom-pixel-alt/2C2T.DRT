"""Optional OpenCL acceleration for large dense matrix multiplications.

The core framework stays NumPy/OpenBLAS-first.  When PyOpenCL and a GPU (an
integrated GPU is fine) are available, the explicit ``opencl``/``auto`` device
mode can offload sufficiently large 2-D GEMMs.  Small matrices stay on the
CPU because queue and host/device transfer costs are larger than the work.
"""

import numpy as np


_KERNEL = r"""
__kernel void matmul(
    const int m, const int n, const int k,
    __global const float *a, __global const float *b, __global float *out) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int index = 0; index < k; ++index) {
        sum += a[row * k + index] * b[index * n + col];
    }
    out[row * n + col] = sum;
}
"""


class _OpenCLMatmul:
    def __init__(self, device):
        import pyopencl as cl

        self.cl = cl
        self.device = device
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, _KERNEL).build()

    def matmul(self, left, right):
        left = np.ascontiguousarray(left, dtype=np.float32)
        right = np.ascontiguousarray(right, dtype=np.float32)
        m, k = left.shape
        _, n = right.shape
        result = np.empty((m, n), dtype=np.float32)
        flags = self.cl.mem_flags
        left_buffer = self.cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=left)
        right_buffer = self.cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=right)
        result_buffer = self.cl.Buffer(self.context, flags.WRITE_ONLY, result.nbytes)
        self.program.matmul(
            self.queue, (m, n), None,
            np.int32(m), np.int32(n), np.int32(k),
            left_buffer, right_buffer, result_buffer,
        )
        self.cl.enqueue_copy(self.queue, result, result_buffer).wait()
        return result


_accelerator = None
_mode = "cpu"
_reason = "CPU/OpenBLAS selected"
# Below this threshold OpenBLAS wins on most laptops once queue overhead is
# included.  It can be adjusted by configure_accelerator for benchmarking.
_min_operations = 32_000_000


def _find_gpu_device():
    try:
        import pyopencl as cl
    except ImportError:
        return None, "PyOpenCL is not installed"
    try:
        devices = [
            device
            for platform in cl.get_platforms()
            for device in platform.get_devices()
            if device.type & cl.device_type.GPU
        ]
    except Exception as error:
        return None, f"OpenCL discovery failed: {error}"
    if not devices:
        return None, "no OpenCL GPU device found"
    # Prefer a unified-memory device when present; this normally selects an
    # integrated GPU and avoids a PCIe round trip on consumer hardware.
    device = next((item for item in devices if getattr(item, "host_unified_memory", False)), devices[0])
    return device, ""


def configure_accelerator(device="auto", min_operations=None):
    """Select ``cpu``, ``auto`` or ``opencl`` execution for supported GEMMs.

    ``auto`` is safe on all systems: it falls back to NumPy when the optional
    OpenCL runtime is unavailable.  ``opencl`` raises a useful error instead
    of silently claiming that a GPU was selected.
    """
    global _accelerator, _mode, _reason, _min_operations
    if device not in ("cpu", "auto", "opencl"):
        raise ValueError("device must be 'cpu', 'auto', or 'opencl'")
    if min_operations is not None:
        if min_operations < 1:
            raise ValueError("min_operations must be positive")
        _min_operations = int(min_operations)
    _accelerator = None
    if device == "cpu":
        _mode, _reason = "cpu", "CPU/OpenBLAS selected by user"
        return accelerator_info()

    selected, reason = _find_gpu_device()
    if selected is None:
        if device == "opencl":
            raise RuntimeError(
                f"OpenCL was requested but is unavailable ({reason}). "
                "Install PyOpenCL and a GPU driver, or use --device cpu."
            )
        _mode, _reason = "cpu", f"CPU fallback: {reason}"
        return accelerator_info()
    try:
        _accelerator = _OpenCLMatmul(selected)
    except Exception as error:
        if device == "opencl":
            raise RuntimeError(f"OpenCL initialization failed: {error}") from error
        _mode, _reason = "cpu", f"CPU fallback: OpenCL initialization failed: {error}"
        return accelerator_info()

    _mode = "opencl"
    _reason = f"OpenCL GPU selected: {selected.name.strip()}"
    return accelerator_info()


def accelerator_info():
    return {
        "mode": _mode,
        "reason": _reason,
        "min_operations": _min_operations,
        "device_name": _accelerator.device.name.strip() if _accelerator else None,
        "integrated": bool(getattr(_accelerator.device, "host_unified_memory", False)) if _accelerator else False,
    }


def matmul(left, right):
    """Run a 2-D GEMM on the selected accelerator when profitable."""
    if (
        _accelerator is None
        or left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or left.shape[0] * left.shape[1] * right.shape[1] < _min_operations
    ):
        return left @ right
    try:
        return _accelerator.matmul(left, right)
    except Exception:
        # Do not interrupt a long CPU training run because an optional driver
        # was reset.  Later calls stay on the known-correct CPU path.
        configure_accelerator("cpu")
        return left @ right
