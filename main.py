"""
c2t.DRT
=========================================================
Train neural networks of ANY size on CPU,
with automatic sharding, multi-core parallelism,
and memory optimization.

Usage:
  python main.py
  python main.py --model large --epochs 100 --batch-size 64
  python main.py --model huge --shard --grad-accum 8
  python main.py --auto-batch --parallel --eval-only
"""

import argparse
import time
import numpy as np

try:
    import c2t

except ImportError:
    import sys, os

    sys.path.insert(0, os.path.dirname(__file__))
    import c2t
from c2t.data import DataLoader
from c2t.memory import (
    get_available_memory_mb,
    suggest_batch_size,
    estimate_training_memory,
)
from c2t.sharding import auto_shard_model
from c2t.parallel import set_num_threads, cpu_count
from c2t.accelerator import configure_accelerator


def load_or_generate_data():
    import os, gzip, struct, urllib.request

    base = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(base, exist_ok=True)

    paths = {
        "train_images": os.path.join(base, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(base, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(base, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(base, "t10k-labels-idx1-ubyte.gz"),
    }

    all_exist = all(os.path.exists(v) for v in paths.values())
    use_synthetic = False

    if not all_exist:
        print("[Data] Downloading MNIST...")
        urls = {
            "train_images": "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        }
        try:
            for key, url in urls.items():
                print(f"  {os.path.basename(url)}")
                urllib.request.urlretrieve(url, paths[key])
            print("[OK] MNIST downloaded")
        except Exception as e:
            print(f"[!] Download failed: {e}")
            print("[Data] Generating synthetic data")
            use_synthetic = True

    if use_synthetic:
        n_train, n_test = 5000, 1000
        x_train = np.random.randn(n_train, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, size=n_train).astype(np.intp)
        x_test = np.random.randn(n_test, 1, 28, 28).astype(np.float32)
        y_test = np.random.randint(0, 10, size=n_test).astype(np.intp)
        print(f"  Synthetic: {n_train} train + {n_test} test")
        return (x_train, y_train), (x_test, y_test)

    def read_images(path, num=None):
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            if num:
                n = min(n, num)
            raw = f.read(n * rows * cols)
            return (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(n, 1, rows, cols)
                .astype(np.float32)
                / 255.0
            )

    def read_labels(path, num=None):
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            if num:
                n = min(n, num)
            raw = f.read(n)
            return np.frombuffer(raw, dtype=np.uint8).astype(np.intp)

    x_train = read_images(paths["train_images"])
    y_train = read_labels(paths["train_labels"])
    x_test = read_images(paths["test_images"])
    y_test = read_labels(paths["test_labels"])

    print(f"  MNIST: {x_train.shape[0]} train + {x_test.shape[0]} test")
    return (x_train, y_train), (x_test, y_test)


def build_model(model_type="mlp"):
    if model_type == "mlp":
        return c2t.Sequential(
            c2t.Flatten(),
            c2t.DenseReLU(784, 512),
            c2t.Dropout(0.3),
            c2t.DenseReLU(512, 256),
            c2t.Dropout(0.2),
            c2t.DenseReLU(256, 128),
            c2t.Dense(128, 10),
        )
    elif model_type == "large":
        return c2t.Sequential(
            c2t.Flatten(),
            c2t.DenseReLU(784, 2048),
            c2t.Dropout(0.3),
            c2t.DenseReLU(2048, 2048),
            c2t.Dropout(0.3),
            c2t.DenseReLU(2048, 1024),
            c2t.Dropout(0.2),
            c2t.DenseReLU(1024, 512),
            c2t.Dense(512, 10),
        )
    elif model_type == "deep":
        layers = [c2t.Flatten()]
        dims = [784, 1024, 1024, 1024, 512, 512, 256, 256, 128, 10]
        for i in range(len(dims) - 2):
            layers.append(c2t.DenseReLU(dims[i], dims[i + 1]))
            layers.append(c2t.Dropout(0.2))
        layers.append(c2t.Dense(dims[-2], dims[-1]))
        return c2t.Sequential(*layers)
    elif model_type == "huge":
        layers = [c2t.Flatten()]
        dims = [784] + [4096] * 6 + [2048] * 4 + [1024] * 2 + [10]
        for i in range(len(dims) - 2):
            layers.append(c2t.DenseReLU(dims[i], dims[i + 1]))
            layers.append(c2t.Dropout(0.2))
        layers.append(c2t.Dense(dims[-2], dims[-1]))
        return c2t.Sequential(*layers)
    elif model_type == "cnn":
        return c2t.Sequential(
            c2t.Conv2DReLU(1, 32, kernel_size=3, padding=1),
            c2t.Conv2DReLU(32, 64, kernel_size=3, padding=1),
            c2t.Flatten(),
            c2t.DenseReLU(64 * 28 * 28, 256),
            c2t.Dropout(0.3),
            c2t.Dense(256, 10),
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="c2t.DRT")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--model", choices=["mlp", "large", "deep", "huge", "cnn"], default="mlp"
    )
    parser.add_argument(
        "--optimizer", choices=["sgd", "adam", "adamw", "rmsprop"], default="adam"
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="recompute Sequential segments during backward to save activation RAM",
    )
    parser.add_argument(
        "--checkpoint-segments",
        type=int,
        default=4,
        help="number of retained checkpoint segments (default: 4)",
    )
    parser.add_argument("--early-stop", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "opencl"],
        default="auto",
        help="use a supported OpenCL GPU for large dense matmuls when available",
    )
    parser.add_argument("--shard", action="store_true", help="enable memory sharding")
    parser.add_argument(
        "--shard-size", type=int, default=200, help="max shard size (MB)"
    )
    parser.add_argument("--auto-batch", action="store_true", help="auto batch sizing")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.grad_accum < 1:
        parser.error("--grad-accum must be at least 1")
    if args.checkpoint_segments < 1:
        parser.error("--checkpoint-segments must be at least 1")
    if args.checkpoint and args.shard:
        parser.error("--checkpoint cannot currently be combined with --shard")

    np.random.seed(args.seed)

    if args.threads > 0:
        set_num_threads(args.threads)
    try:
        accelerator = configure_accelerator(args.device)
    except RuntimeError as error:
        parser.error(str(error))

    print("=" * 68)
    print("  c2t.DRT")
    print("  Deep Learning for CPU : models of ANY size")
    print("=" * 68)
    print(f"  CPU cores: {cpu_count()} | Threads: {args.threads or 'auto'}")
    print(f"  Compute device: {accelerator['mode']} ({accelerator['reason']})")
    print(f"  RAM available: {get_available_memory_mb():.0f} MB")
    print(f"  Model: {args.model} | Optimizer: {args.optimizer} | LR: {args.lr}")
    print(f"  Batch: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(
        f"  Checkpointing: {'ON (' + str(args.checkpoint_segments) + ' segments)' if args.checkpoint else 'OFF'}"
    )
    print(
        f"  Sharding: {'ON (' + str(args.shard_size) + 'MB/shard)' if args.shard else 'OFF'}"
    )
    print()

    print("[1/5] Loading data...")
    (x_train_raw, y_train_raw), (x_test, y_test) = load_or_generate_data()

    val_size = int(len(x_train_raw) * args.val_split)
    train_size = len(x_train_raw) - val_size
    print(f"\n[2/5] Split train/val: {train_size}/{val_size}")

    indices = np.random.permutation(len(x_train_raw))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    x_train, y_train = x_train_raw[train_idx], y_train_raw[train_idx]
    x_val, y_val = x_train_raw[val_idx], y_train_raw[val_idx]

    train_data = c2t.data.TensorDataset(x_train, y_train)
    val_data = c2t.data.TensorDataset(x_val, y_val)
    test_data = c2t.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print(f"\n[3/5] Building model {args.model}...")
    model = build_model(args.model)

    if args.load:
        state = np.load(args.load, allow_pickle=True)
        model.load_state_dict(state["model_state"].item())
        print(f"  Model loaded from {args.load}")

    if args.shard:
        from c2t.sharding import ShardedModel

        model = ShardedModel(model, max_shard_size_mb=args.shard_size)

    opt_map = {
        "sgd": c2t.SGD,
        "adam": c2t.Adam,
        "adamw": c2t.AdamW,
        "rmsprop": c2t.RMSprop,
    }
    optimizer = opt_map[args.optimizer](model.parameters(), lr=args.lr)

    loss_fn = c2t.CrossEntropyLoss()
    scheduler = c2t.LRScheduler(
        optimizer, factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    trainer = c2t.Trainer(model, loss_fn, optimizer, verbose=True)

    print(f"\n  Model summary:")
    total_params = trainer.summary()
    sample_shape = (
        (784,) if args.model in ("mlp", "large", "deep", "huge") else (1, 28, 28)
    )
    est = estimate_training_memory(
        model, sample_shape, batch_size=args.batch_size, optimizer=optimizer
    )
    print(
        f"  Estimated RAM needed: ~{est['total_megabytes']:.0f} MB "
        f"(params {est['parameters_megabytes']:.0f} + activations {est['activations_megabytes']:.0f} "
        f"+ optimizer {est['optimizer_megabytes']:.0f})"
    )
    print()

    if args.eval_only:
        print("[Evaluation]...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        return

    print(f"\n[4/5] Starting training...")
    print(f"  Sharding: {'Active' if args.shard else 'Inactive'}")
    print(f"  Auto-batch: {'Active' if args.auto_batch else 'Inactive'}")
    print(f"  Checkpointing: {'Active' if args.checkpoint else 'Inactive'}")
    print()

    t_start = time.time()

    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        grad_accumulation=args.grad_accum,
        early_stopping_patience=args.early_stop if args.early_stop > 0 else None,
        lr_scheduler=scheduler,
        save_path=args.save,
        verbose_interval=1,
        auto_batch=args.auto_batch,
        gradient_checkpointing=args.checkpoint,
        checkpoint_segments=args.checkpoint_segments,
    )

    t_total = time.time() - t_start

    print()
    print("=" * 68)
    print("  FINAL RESULTS")
    print("=" * 68)
    print(f"  Total time: {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"  Epochs: {len(history['train_loss'])}")

    final_train_loss = history["train_loss"][-1]
    final_train_acc = history["train_acc"][-1]
    print(f"  Train Loss: {final_train_loss:.4f} | Train Acc: {final_train_acc:.4f}")

    if history["val_loss"]:
        best_val_idx = int(np.argmin(history["val_loss"]))
        print(
            f"  Best Val Loss: {history['val_loss'][best_val_idx]:.4f} "
            f"(epoch {best_val_idx + 1}) | "
            f"Val Acc: {history['val_acc'][best_val_idx]:.4f}"
        )

    print(f"\n[5/5] Final evaluation...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    sample = x_test[0:4]
    preds = trainer.predict(sample)
    pred_classes = np.argmax(preds, axis=1)
    print(f"\n  Inference example:")
    print(f"    Pred: {pred_classes.tolist()}")
    print(f"    True: {y_test[0:4].tolist()}")

    print()
    print("  [OK] c2t.DRT training complete !")
    print("=" * 68)


if __name__ == "__main__":
    main()
