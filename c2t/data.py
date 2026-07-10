import numpy as np
import os
from .tensor import Tensor


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert len(tensors) > 0, "at least one tensor required"
        self.tensors = tensors
        self.n = len(tensors[0])
        for t in tensors:
            assert len(t) == self.n, "all tensors must have same length"

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return tuple(t[idx] if isinstance(t, (np.ndarray, Tensor)) else t[idx].data for t in self.tensors)


class MemoryMapDataset(Dataset):
    def __init__(self, filename, shape, dtype=np.float32, mode='r'):
        self.filename = filename
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = dtype
        self.mode = mode
        self._mmap = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self._mmap[idx].copy()

    def flush(self):
        self._mmap.flush()

    def close(self):
        del self._mmap

    @staticmethod
    def from_array(filename, array, mode='w+'):
        mmap = np.memmap(filename, dtype=array.dtype, mode=mode, shape=array.shape)
        mmap[:] = array[:]
        mmap.flush()
        return MemoryMapDataset(filename, array.shape, array.dtype, mode='r')


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False, prefetch=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = len(dataset)
        self.indices = np.arange(self.n)

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self.n:
            raise StopIteration
        end = min(self._idx + self.batch_size, self.n)
        if self.drop_last and end - self._idx < self.batch_size:
            raise StopIteration
        batch_indices = self.indices[self._idx:end]
        self._idx = end
        batch_items = [self.dataset[i] for i in batch_indices]
        n_tensors = len(batch_items[0])
        batched = []
        for j in range(n_tensors):
            items = [item[j] for item in batch_items]
            if isinstance(items[0], (int, np.integer)):
                batched.append(np.array(items, dtype=np.int64))
            elif isinstance(items[0], float):
                batched.append(np.array(items, dtype=np.float32))
            else:
                batched.append(np.stack(items, axis=0).astype(np.float32))
        return tuple(batched)


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    indices = np.random.permutation(len(dataset))
    splits = []
    start = 0
    for l in lengths:
        splits.append(Subset(dataset, indices[start:start + l]))
        start += l
    return splits


class TensorDatasetFromArrays(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.n = len(data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        if self.labels is not None:
            return x, self.labels[idx]
        return x


def normalize(data, mean=None, std=None):
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0) + 1e-8
    return (data - mean) / std, mean, std


class DataAugmentation:
    @staticmethod
    def random_noise(x, noise_level=0.01):
        return x + np.random.randn(*x.shape) * noise_level

    @staticmethod
    def random_shift(x, shift_range=2):
        h, w = x.shape[-2:]
        shift_h = np.random.randint(-shift_range, shift_range + 1)
        shift_w = np.random.randint(-shift_range, shift_range + 1)
        return np.roll(x, (shift_h, shift_w), axis=(-2, -1))

    @staticmethod
    def random_flip(x, axis=-1):
        if np.random.random() < 0.5:
            return np.flip(x, axis=axis).copy()
        return x
