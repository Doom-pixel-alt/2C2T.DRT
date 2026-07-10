import numpy as np
import os
import gc
import tempfile
from .tensor import Tensor, no_grad
from .memory import free_memory, estimate_model_size


class ModelShard:
    def __init__(self, layers, shard_id=0):
        self.layers = layers
        self.shard_id = shard_id
        self._cache_dir = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.get_parameters():
            p.zero_grad()

    def to_disk(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'state_dict'):
                state[f"layer_{i}"] = layer.state_dict()
        np.savez_compressed(path, **state)

    @staticmethod
    def from_disk(path, layer_factories):
        loaded = np.load(path, allow_pickle=True)
        layers = []
        for i, factory in enumerate(layer_factories):
            layer = factory()
            key = f"layer_{i}"
            if key in loaded:
                layer.load_state_dict(loaded[key].item())
            layers.append(layer)
        return ModelShard(layers)


class ShardedModel:
    def __init__(self, model, max_shard_size_mb=500, temp_dir=None, offload_freq='epoch'):
        self.model = model
        self.max_shard_size_mb = max_shard_size_mb
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), "c2t_shards")
        self.offload_freq = offload_freq
        self.shards = []
        self._active_shard = None
        self._shard_offload_paths = {}
        self._param_shapes = {}
        self._build_shards()

    def _build_shards(self):
        if not hasattr(self.model, '_modules') or not self.model._modules:
            self.shards = [ModelShard([self.model])]
            return

        layers = list(self.model._modules.values())
        current_shard = []
        current_size = 0

        for layer in layers:
            layer_size = estimate_model_size(layer)['megabytes']
            if current_size + layer_size > self.max_shard_size_mb and current_shard:
                self.shards.append(ModelShard(current_shard, len(self.shards)))
                current_shard = []
                current_size = 0
            current_shard.append(layer)
            current_size += layer_size

        if current_shard:
            self.shards.append(ModelShard(current_shard, len(self.shards)))

        print(f"[Sharding] Modele divise en {len(self.shards)} shards "
              f"(max {self.max_shard_size_mb} MB/shard)")

    def _save_shard(self, shard_id):
        """Save shard params to a file in temp dir."""
        path = os.path.join(self.temp_dir, f"shard_{shard_id}.npz")
        os.makedirs(self.temp_dir, exist_ok=True)
        state = {}
        shard = self.shards[shard_id]
        for i, layer in enumerate(shard.layers):
            if hasattr(layer, 'state_dict'):
                sub = layer.state_dict()
                for k, v in sub.items():
                    state[f"layer_{i}_{k}"] = v
        np.savez_compressed(path, **state)
        self._shard_offload_paths[shard_id] = path

    def _free_shard(self, shard_id):
        """Release parameter memory for a shard."""
        shard = self.shards[shard_id]
        for p in shard.get_parameters():
            self._param_shapes[id(p)] = p.data.shape
            p.data = np.zeros((1,), dtype=np.float32)
            p.grad = None

    def _restore_shard(self, shard_id):
        """Restore parameters from disk for a shard."""
        shard = self.shards[shard_id]
        path = self._shard_offload_paths.get(shard_id)
        if not path or not os.path.exists(path):
            return shard
        with np.load(path, allow_pickle=True) as loaded:
            for p in shard.get_parameters():
                shape = self._param_shapes.get(id(p))
                if shape:
                    p.data = np.zeros(shape, dtype=np.float32)
            for i, layer in enumerate(shard.layers):
                if hasattr(layer, 'state_dict'):
                    prefix = f"layer_{i}_"
                    sub = {}
                    for k in list(loaded.keys()):
                        if k.startswith(prefix):
                            sub[k[len(prefix):]] = loaded[k]
                    if sub:
                        layer.load_state_dict(sub)
        self._shard_offload_paths.pop(shard_id, None)
        try:
            os.remove(path)
        except:
            pass
        return shard

    def load_all(self):
        """Load all shards into RAM (for training)."""
        for i in range(len(self.shards)):
            self._restore_shard(i)
        self._active_shard = -1

    def offload_all(self):
        """Save all shards to disk and free RAM."""
        for i in range(len(self.shards)):
            if i not in self._shard_offload_paths:
                self._save_shard(i)
                self._free_shard(i)
        self._active_shard = None
        free_memory()

    def forward(self, x):
        self.load_all()
        for shard in self.shards:
            x = shard.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def step_end(self):
        """Call after optimizer.step() to free memory between steps.
        Frequency is self.offload_freq: 'never', 'epoch', or 'step'."""
        if getattr(self, 'offload_freq', 'epoch') == 'step':
            self.offload_all()

    def epoch_end(self):
        """Call at end of epoch to free memory."""
        if getattr(self, 'offload_freq', 'epoch') in ('epoch', 'step'):
            self.offload_all()

    def zero_grad(self):
        for shard in self.shards:
            shard.zero_grad()

    def parameters(self):
        params = []
        for shard in self.shards:
            params.extend(shard.get_parameters())
        return params

    def named_parameters(self):
        named = []
        for sid, shard in enumerate(self.shards):
            for layer in shard.layers:
                if hasattr(layer, 'named_parameters'):
                    try:
                        named.extend(layer.named_parameters())
                    except:
                        for k, v in layer._parameters.items():
                            named.append((f"shard{sid}_{k}", v))
                else:
                    for k, v in getattr(layer, '_parameters', {}).items():
                        named.append((f"shard{sid}_{k}", v))
        return named

    def state_dict(self):
        state = {}
        for sid, shard in enumerate(self.shards):
            for j, layer in enumerate(shard.layers):
                if hasattr(layer, 'state_dict'):
                    sub_state = layer.state_dict()
                    for k, v in sub_state.items():
                        state[f"shard{sid}_layer{j}_{k}"] = v
        return state

    def load_state_dict(self, state_dict):
        for sid, shard in enumerate(self.shards):
            for j, layer in enumerate(shard.layers):
                if hasattr(layer, 'load_state_dict'):
                    prefix = f"shard{sid}_layer{j}_"
                    sub_state = {}
                    for k, v in state_dict.items():
                        if k.startswith(prefix):
                            sub_state[k[len(prefix):]] = v
                    if sub_state:
                        layer.load_state_dict(sub_state)

    def train(self, mode=True):
        for shard in self.shards:
            for layer in shard.layers:
                if hasattr(layer, 'train'):
                    layer.train(mode)

    def eval(self):
        self.train(False)

    def extra_repr(self):
        return f"shards={len(self.shards)}, max_shard={self.max_shard_size_mb}MB"

    def __repr__(self):
        return f"ShardedModel({len(self.shards)} shards)"


def auto_shard_model(model, memory_limit_mb=None, model_size_mb=None):
    if memory_limit_mb is None:
        import psutil
        try:
            memory_limit_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.5
        except:
            memory_limit_mb = 500

    if model_size_mb is None:
        model_size_mb = estimate_model_size(model)['megabytes']

    if model_size_mb < memory_limit_mb * 0.8:
        return model

    max_shard = max(64, int(memory_limit_mb * 0.3))
    print(f"[AutoShard] Modele {model_size_mb:.0f}MB, RAM dispo ~{memory_limit_mb:.0f}MB")
    print(f"[AutoShard] Sharding en blocs de {max_shard}MB...")
    return ShardedModel(model, max_shard_size_mb=max_shard)


def offload_optimizer_state(optimizer, path_prefix, keep_in_ram=False):
    path = f"{path_prefix}_opt_state.npz"
    state = {}
    for attr in ['_m', '_v', '_sq', '_velocities', '_buf']:
        if hasattr(optimizer, attr):
            arr = getattr(optimizer, attr)
            if arr and isinstance(arr, list) and len(arr) > 0:
                for i, a in enumerate(arr):
                    if a is not None:
                        state[f"{attr}_{i}"] = a
    np.savez_compressed(path, **state)

    if not keep_in_ram:
        for attr in ['_m', '_v', '_sq', '_velocities', '_buf']:
            if hasattr(optimizer, attr):
                arr = getattr(optimizer, attr)
                if arr and isinstance(arr, list):
                    for i in range(len(arr)):
                        if arr[i] is not None:
                            arr[i] = np.array([0.0], dtype=np.float32)
    return path
