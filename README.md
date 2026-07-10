<div align="center">

# 2C2T.DRT

**2C2T.DRT**

*Train neural networks on CPU. No GPU. No budget. No bullshit.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dependencies: numpy only](https://img.shields.io/badge/dependencies-numpy%20only-orange)](https://numpy.org)

</div>

---

## Why 2C2T.DRT ?

GPUs needed for deep learning have become unaffordable. An RTX 3060 costs **500 €+** when available. RTX 4090s at **2500 €+** are perpetually out of stock. Training AI models has become a luxury.

**2C2T.DRT breaks this lock.** A 100% CPU framework, zero GPU dependencies, it lets you train real neural networks on any computer. No graphics card needed. No expensive cloud. No 32 GB of VRAM required.

---

## What it does (honestly)

Real benchmarks, measured on a standard 16-core CPU :

| Model | Size | Batch | Time/step | Samples/s |
|-------|------|-------|-----------|-----------|
| MLP 4 layers (568K params) | 2.2 MB | 64 | **50 ms** | 1282 |
| MLP 5 layers (8.4M params) | 32 MB | 16 | **393 ms** | 41 |
| MLP 4 layers (32M params) | 108 MB | 8 | **1049 ms** | 8 |

> **Note**: These measurements include full forward + backward + Adam update. Not raw BLAS benchmarks.

### In practice: one MNIST epoch (60K samples)

- 568K params model, batch 64 → **~47 seconds** per epoch
- 10 epochs → **~8 minutes**

It's slow compared to a GPU (15-30 seconds), but it **works on any PC**.

---

## Installation

```bash
git clone https://github.com/Doom-pixel-alt/2C2T.DRT.git
cd 2C2T.DRT
python main.py
```

Single dependency: **NumPy** (uses OpenBLAS, assembly-optimized for matrix operations).

---

## Usage

### Python

```python
import c2t as nn
from c2t.data import DataLoader, TensorDataset

# 1. Build the model
model = nn.Sequential(
    nn.Flatten(),
    nn.DenseReLU(784, 512),    # Fused Dense + ReLU (1 autograd node)
    nn.DenseReLU(512, 256),
    nn.Dense(256, 10),
)

# 2. Optimizer + loss function
optimizer = nn.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 3. Trainer
trainer = nn.Trainer(model, loss_fn, optimizer)
trainer.fit(train_loader, val_loader, epochs=10)

# 4. Inference
predictions = trainer.predict(x_test)
```

### Command line

```bash
# MNIST (or synthetic data if offline)
python main.py

# Large model with memory sharding
python main.py --model large --shard --shard-size 200

# Long training run
python main.py --model deep --epochs 100 --lr 0.0005 --grad-accum 4

# Evaluation only
python main.py --model cnn --eval-only --load model.npz

# Benchmark
python benchmark.py
```

---

## Available architectures

| Flag | Model | Size | Parameters | RAM needed |
|------|-------|------|------------|------------|
| `mlp` | 4 layer fully connected | 2.2 MB | 567 K | Very low |
| `large` | 5 layer fully connected | 32 MB | 8.4 M | Low |
| `deep` | 9 layer fully connected | 26 MB | 6.7 M | Low |
| `huge` | 12 layer fully connected | 860 MB | 226 M | High |
| `cnn` | Conv2D + fully connected | 137 MB | 36 M | High |

---

## Features

### Core

| Feature | Status |
|---------|--------|
| Tensor with autograd (20+ differentiable ops) | ✅ Stable |
| Layers: Dense, Conv2D, BatchNorm, Dropout | ✅ Stable |
| Activations: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax | ✅ Stable |
| Fused DenseReLU (Dense + ReLU in 1 node) | ✅ Stable |
| Optimizers: SGD, Adam, AdamW, RMSprop | ✅ Stable |
| Loss: MSE, MAE, CrossEntropy, BinaryCE, Huber, NLL | ✅ Stable |

### Memory optimization

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **Gradient accumulation** | Accumulate gradients over N micro-batches | Simulate larger batches without extra RAM |
| **Gradient checkpointing** | Recomputed activations on backward | Trade time for memory (50% less RAM) |
| **Memory sharding** | Split model, load/unload from disk | Models larger than RAM |
| **Quantization 8/16-bit** | Weight compression | Up to 4× less RAM/storage |
| **Auto-batch** | Compute optimal batch size | Avoid MemoryError |

### Training

| Feature | Status |
|---------|--------|
| Full trainer with metrics | ✅ Stable |
| LR Scheduler (ReduceLROnPlateau) | ✅ Stable |
| Early Stopping | ✅ Stable |
| Save / Load weights | ✅ Stable |
| DataLoader with shuffle + batching | ✅ Stable |
| MemoryMapDataset (files > RAM) | ✅ Stable |

### Infrastructure

| Feature | Status |
|---------|--------|
| OpenBLAS multi-core parallelism | ✅ Stable |
| mmap storage for streaming inference | ✅ Stable |
| Cross-platform (Windows, Linux, macOS) | ✅ Tested (Windows 11) |
| Zero GPU dependency | ✅ Guaranteed |

---

## Known limitations (honestly)

- **No miracle**: expect 15-30× slower than an entry-level GPU
- **RAM bound**: model size limited by available RAM (not VRAM)
- **No 100B+ training**: backward pass activations must fit in RAM. Streaming only works for inference.
- **No "custom" kernels**: numpy/OpenBLAS are already assembly-optimized. Our matmuls will never beat theirs.

### When to use it

| Use case | Recommendation |
|----------|---------------|
| You have a GPU | **Use PyTorch**. 2C2T.DRT isn't for you. |
| You DON'T have a GPU | **2C2T.DRT is perfect**. Learn, prototype, train. |
| You want to learn deep learning | **Start here**. No cloud, no complex setup. |
| You need CPU deployment | **Ideal**. No CUDA dependency, single package. |
| Your model is too big for VRAM | **Sharding**: load/unload weights from disk. |

---

## GPU vs CPU comparison (real 2024-2025 prices)

| Solution | Price (€) | Raw perf | Availability |
|----------|-----------|----------|--------------|
| **2C2T.DRT (CPU)** | **0 €** (you already own a PC) | 1× | ✅ Immediate |
| RTX 3060 12 GB | 450-550 € (shortage) | 30× | ⚠️ Frequent stockout |
| RTX 4060 Ti 16 GB | 600-700 € | 40× | ⚠️ Limited stock |
| RTX 4090 24 GB | 2500-3500 € (scalping) | 100× | ❌ Perpetual shortage |
| A100 80 GB (pro) | 25 000-35 000 € | 300× | ❌ Enterprise only |
| Cloud GPU (rental) | 1-5 €/h | Variable | ✅ But expensive long-term |

**Bottom line**: If you own a PC, you already have everything you need for deep learning with 2C2T.DRT. It's slow, but it's free and it works.

---

## Project structure

```
├── c2t/                        # Python package
│   ├── __init__.py             # Public API
│   ├── tensor.py               # Tensor with autograd
│   ├── layers/__init__.py      # All layers
│   ├── optimizers.py           # Optimizers
│   ├── losses.py               # Loss functions
│   ├── trainer.py              # Training engine
│   ├── data.py                 # Data loading
│   ├── parallel.py             # CPU parallelism
│   ├── sharding.py             # Memory sharding
│   ├── storage.py              # mmap storage
│   └── memory.py               # Memory optimization
├── main.py                     # CLI entry point
├── benchmark.py                # Benchmarks
└── README.md
```

---

## License

MIT — do whatever you want. Improve it, fork it, distribute it.


