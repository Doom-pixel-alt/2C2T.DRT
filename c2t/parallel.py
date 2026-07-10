import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

_NUM_THREADS = min(mp.cpu_count(), 2)
_BACKEND = "numpy"


def cpu_count():
    return mp.cpu_count()


def set_num_threads(n):
    global _NUM_THREADS
    _NUM_THREADS = max(1, min(n, mp.cpu_count()))
    os.environ["OMP_NUM_THREADS"] = str(_NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(_NUM_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_NUM_THREADS)


def get_num_threads():
    return _NUM_THREADS


def parallel_map(fn, items, max_workers=None):
    if max_workers is None:
        max_workers = _NUM_THREADS
    if len(items) == 1 or max_workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items))


def parallel_batch_apply(fn, batch, axis=0, n_splits=None):
    if n_splits is None:
        n_splits = _NUM_THREADS
    if n_splits <= 1 or len(batch) < n_splits:
        return fn(batch)
    splits = np.array_split(batch, n_splits, axis=axis)
    results = parallel_map(fn, splits, max_workers=n_splits)
    return np.concatenate(results, axis=axis)


def matmul_parallel(A, B, threshold=512):
    M, K = A.shape
    K2, N = B.shape
    if M < threshold or N < threshold or _NUM_THREADS <= 1:
        return A @ B
    splits = np.array_split(A, _NUM_THREADS, axis=0)
    def _mul(chunk):
        return chunk @ B
    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        results = list(pool.map(_mul, splits))
    return np.vstack(results)


def element_wise_parallel(fn, arr, axis=0, threshold=1024):
    if arr.shape[axis] < threshold or _NUM_THREADS <= 1:
        return fn(arr)
    splits = np.array_split(arr, _NUM_THREADS, axis=axis)
    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        results = list(pool.map(fn, splits))
    return np.concatenate(results, axis=axis)


def parallel_conv2d(x, weight, stride=1, padding=0):
    if _NUM_THREADS <= 1 or x.shape[0] < _NUM_THREADS:
        return _conv2d_single(x, weight, stride, padding)
    splits = np.array_split(x, _NUM_THREADS, axis=0)
    fn = partial(_conv2d_single, weight=weight, stride=stride, padding=padding)
    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        results = list(pool.map(fn, splits))
    return np.concatenate(results, axis=0)


def _conv2d_single(x, weight, stride, padding):
    N, C, H, W = x.shape
    O, _, KH, KW = weight.shape
    SH, SW = stride if isinstance(stride, tuple) else (stride, stride)
    PH, PW = padding if isinstance(padding, tuple) else (padding, padding)
    if PH > 0 or PW > 0:
        x = np.pad(x, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    cols = np.zeros((N, C, KH, KW, OH, OW), dtype=np.float32)
    for i in range(KH):
        for j in range(KW):
            cols[:, :, i, j, :, :] = x[:, :, i:i + OH * SH:SH, j:j + OW * SW:SW]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, C * KH * KW)
    w_cols = weight.reshape(O, -1)
    out = (cols @ w_cols.T).reshape(N, OH, OW, O).transpose(0, 3, 1, 2)
    return out


def adam_parallel(params, grads, m, v, lr, b1, b2, eps, step):
    if len(params) <= 1 or _NUM_THREADS <= 1:
        for i in range(len(params)):
            m[i] = b1 * m[i] + (1 - b1) * grads[i]
            v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
            m_hat = m[i] / (1 - b1 ** step)
            v_hat = v[i] / (1 - b2 ** step)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return params, m, v

    def _update(args):
        i, p, g, mi, vi = args
        mi = b1 * mi + (1 - b1) * g
        vi = b2 * vi + (1 - b2) * g ** 2
        m_hat = mi / (1 - b1 ** step)
        v_hat = vi / (1 - b2 ** step)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return i, p, mi, vi

    items = [(i, params[i].copy(), grads[i], m[i].copy(), v[i].copy())
             for i in range(len(params))]
    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        results = list(pool.map(_update, items))
    for i, p, mi, vi in results:
        params[i] = p
        m[i] = mi
        v[i] = vi
    return params, m, v
