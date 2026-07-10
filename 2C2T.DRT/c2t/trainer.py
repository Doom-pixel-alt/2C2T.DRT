import numpy as np
import time
import os
from .tensor import Tensor, no_grad
from .optimizers import LRScheduler
from .memory import free_memory, get_available_memory_mb, suggest_batch_size
from .parallel import get_num_threads, parallel_batch_apply


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.verbose:
                print(f"  Early stopping triggered after {self.wait} epochs without improvement")
            return True
        return False

    def reset(self):
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0


class Trainer:
    def __init__(self, model, loss_fn, optimizer, verbose=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.verbose = verbose
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_val_loss = float('inf')
        self.grad_acc_steps = 0
        self._epoch_times = []
        self._num_workers = get_num_threads()

    def _compute_accuracy(self, pred, target):
        if pred.ndim > 1 and pred.shape[-1] > 1:
            pred_class = np.argmax(pred, axis=-1)
        else:
            pred_class = (pred > 0.5).astype(np.int64)
        if target.ndim > 1:
            target_class = np.argmax(target, axis=-1) if target.shape[-1] > 1 else target
        else:
            target_class = target
        return (pred_class == target_class).mean()

    def train_epoch(self, train_loader, grad_accumulation=1, max_batches=None):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            x_data, y_data = batch[0], batch[1]
            x = Tensor(x_data)
            y = Tensor(y_data)

            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss_val = loss.data.item()

            loss.backward()

            total_loss += loss_val
            total_acc += self._compute_accuracy(pred.data, y_data)
            n_batches += 1

            if (batch_idx + 1) % grad_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if hasattr(self.model, 'step_end'):
                    self.model.step_end()

        if n_batches > 0 and (batch_idx + 1) % grad_accumulation != 0:
            self.optimizer.step()
            if hasattr(self.model, 'step_end'):
                self.model.step_end()

        return total_loss / n_batches, total_acc / n_batches if n_batches else 0

    @no_grad()
    def evaluate(self, data_loader, max_batches=None):
        self.model.eval()
        if hasattr(self.model, 'load_all'):
            self.model.load_all()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break

            x_data, y_data = batch[0], batch[1]
            x = Tensor(x_data)
            y = Tensor(y_data)

            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            total_loss += loss.data.item()
            total_acc += self._compute_accuracy(pred.data, y_data)
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches if n_batches else 0

    def fit(self, train_loader, val_loader=None, epochs=10, grad_accumulation=1,
            early_stopping_patience=None, lr_scheduler=None, save_path=None,
            max_batches_per_epoch=None, verbose_interval=1, auto_batch=False):

        if auto_batch:
            sample = next(iter(train_loader))[0][0].shape
            suggested = suggest_batch_size(self.model, sample)
            if suggested < train_loader.batch_size:
                print(f"[AutoBatch] Batch size ajustee: {train_loader.batch_size} -> {suggested}")
                train_loader.batch_size = suggested

        early_stopping = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None
        self.grad_acc_steps = grad_accumulation

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self.train_epoch(train_loader, grad_accumulation, max_batches_per_epoch)

            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, max_batches_per_epoch)

            self.history["train_loss"].append(float(train_loss))
            self.history["train_acc"].append(float(train_acc))
            if val_loader:
                self.history["val_loss"].append(float(val_loss))
                self.history["val_acc"].append(float(val_acc))

            epoch_time = time.time() - t0
            self._epoch_times.append(epoch_time)

            free_memory()

            if self.verbose and epoch % verbose_interval == 0:
                log = f"Epoch {epoch}/{epochs} | {epoch_time:.1f}s | "
                log += f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
                if val_loader:
                    log += f" | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                print(log)

            if lr_scheduler:
                monitor_loss = val_loss if val_loader else train_loss
                lr_scheduler.step(monitor_loss)

            if early_stopping and val_loader:
                if early_stopping.step(val_loss):
                    print(f"  Best val loss: {early_stopping.best_loss:.4f}")
                    break

            if save_path and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save(save_path)
                if self.verbose:
                    print(f"  Model saved to {save_path} (val_loss: {val_loss:.4f})")

        return self.history

    def save(self, path, save_optimizer=True):
        state = {
            "model_state": self.model.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        if save_optimizer:
            state["optimizer_state"] = self.optimizer.state_dict()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(path, **state)

    def load(self, path):
        state = np.load(path, allow_pickle=True)
        self.model.load_state_dict(state["model_state"].item())
        self.history = state["history"].item()
        self.best_val_loss = float(state["best_val_loss"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"].item())
        print(f"Model loaded from {path}")

    def predict(self, x):
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        with no_grad():
            return self.model(x).data

    def summary(self):
        from .memory import estimate_model_size
        total_params = 0
        for name, param in self.model.named_parameters():
            n = param.data.size
            total_params += n
            print(f"  {name}: {tuple(param.shape)} ({n:,} params)")
        est = estimate_model_size(self.model)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: {est['megabytes']:.2f} MB")
        print(f"  CPU threads: {self._num_workers}")
        if self._epoch_times:
            avg_time = np.mean(self._epoch_times)
            print(f"  Avg epoch time: {avg_time:.2f}s")
        return total_params
