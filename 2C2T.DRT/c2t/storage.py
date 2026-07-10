import numpy as np
import os
import gc
import tempfile
from .memory import free_memory

"""
LIMITES CLAIRES :
- ParameterStore permet de stocker des parametres sur disque (mmap)
- Utile pour l'INFERENCE de gros modeles qui ne tiennent pas en RAM
- NE PERMET PAS l'entrainement en streaming :
  le backward a besoin de TOUTES les activations du forward
  donc les parametres doivent rester charges pendant forward+backward
- Pour l'entrainement de modeles >RAM, voir le gradient checkpointing
  dans memory.py (echange temps <-> memoire)
"""

class ParameterStore:
    def __init__(self, base_dir=None, max_ram_mb=512):
        self.base_dir = base_dir or os.path.join(tempfile.gettempdir(), "c2t_storage")
        self.max_ram_mb = max_ram_mb
        self._current_ram_mb = 0
        self._params = {}
        self._mmap_files = {}
        self._param_index = {}
        os.makedirs(self.base_dir, exist_ok=True)

    def store(self, name, data, persistent=True):
        if persistent and data.nbytes > self.max_ram_mb * 1024 * 1024:
            safe_name = name.replace('/', '_').replace('.', '_')
            path = os.path.join(self.base_dir, f"{safe_name}.dat")
            mmap = np.memmap(path, dtype=data.dtype, mode='w+', shape=data.shape)
            mmap[:] = data[:]
            mmap.flush()
            self._mmap_files[name] = (path, data.shape, data.dtype)
            self._param_index[name] = ('mmap', path)
            del mmap
            return
        else:
            self._params[name] = data.copy()
            self._current_ram_mb += data.nbytes / (1024 * 1024)
            self._param_index[name] = ('ram',)
        if self._current_ram_mb > self.max_ram_mb * 0.8:
            self._evict_oldest()

    def load(self, name):
        if name in self._params:
            return self._params[name]
        if name in self._mmap_files:
            path, shape, dtype = self._mmap_files[name]
            return np.array(np.memmap(path, dtype=dtype, mode='r', shape=shape))
        raise KeyError(f"Parameter '{name}' not found")

    def _evict_oldest(self):
        ram_keys = [k for k in self._params.keys()]
        to_free = int(len(ram_keys) * 0.3)
        for k in ram_keys[:to_free]:
            data = self._params.pop(k)
            safe_name = k.replace('/', '_').replace('.', '_')
            path = os.path.join(self.base_dir, f"{safe_name}.dat")
            mmap = np.memmap(path, dtype=data.dtype, mode='w+', shape=data.shape)
            mmap[:] = data[:]
            mmap.flush()
            self._mmap_files[k] = (path, data.shape, data.dtype)
            self._param_index[k] = ('mmap', path)
            self._current_ram_mb -= data.nbytes / (1024 * 1024)
            del data, mmap
        free_memory()

    def total_size_mb(self):
        total = self._current_ram_mb
        for name in self._mmap_files:
            path, shape, dtype = self._mmap_files[name]
            total += np.prod(shape) * np.dtype(dtype).itemsize / (1024 * 1024)
        return total

    def clear(self):
        self._params.clear()
        for name, (path, _, _) in self._mmap_files.items():
            try: os.remove(path)
            except: pass
        self._mmap_files.clear()
        self._param_index.clear()
        self._current_ram_mb = 0
        free_memory()


class StreamingInferenceModel:
    """
    Charge les couches une par une depuis le disque.
    UTILE POUR L'INFERENCE SEULEMENT.
    Pas de backward, pas de gradients.
    """
    def __init__(self, layer_builders, param_store=None, max_ram_mb=512):
        self.layer_builders = layer_builders
        self.param_store = param_store or ParameterStore(max_ram_mb=max_ram_mb)
        self._current_layer = None
        self._current_layer_idx = -1
        self._layer_count = len(layer_builders)

    def _load_layer(self, idx):
        if idx == self._current_layer_idx and self._current_layer is not None:
            return self._current_layer
        if self._current_layer is not None:
            self._unload_current()
        builder = self.layer_builders[idx]
        layer = builder()
        prefix = f"layer_{idx}"
        for p_name in ['weight', 'bias']:
            if hasattr(layer, p_name):
                param = getattr(layer, p_name)
                stored_name = f"{prefix}_{p_name}"
                try:
                    stored = self.param_store.load(stored_name)
                    param.data[:] = stored
                except KeyError:
                    self.param_store.store(stored_name, param.data.copy())
        self._current_layer = layer
        self._current_layer_idx = idx
        return layer

    def _unload_current(self):
        if self._current_layer is None:
            return
        prefix = f"layer_{self._current_layer_idx}"
        for p_name in ['weight', 'bias']:
            if hasattr(self._current_layer, p_name):
                param = getattr(self._current_layer, p_name)
                stored_name = f"{prefix}_{p_name}"
                self.param_store.store(stored_name, param.data)
                param.data = np.zeros((1,), dtype=np.float32)
        self._current_layer = None
        self._current_layer_idx = -1
        free_memory()

    def forward(self, x):
        self._current_layer = None
        for i in range(self._layer_count):
            layer = self._load_layer(i)
            x = layer(x)
            if i < self._layer_count - 1:
                self._unload_current()
        return x

    def __call__(self, x):
        return self.forward(x)
