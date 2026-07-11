import numpy as np
from .tensor import Tensor, no_grad

def gradient_check(model, loss_fn, x_data, y_data, epsilon=1e-3, tol=0.1, verbose=False):
    model.train()
    x = Tensor(x_data)
    y = Tensor(y_data)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    max_err = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        analytical = param.grad.ravel()
        numerical = np.zeros(min(len(analytical), 20))
        orig = param.data.copy()

        for i in range(len(numerical)):
            param.data.flat[i] = orig.flat[i] + epsilon
            loss_plus = loss_fn(model(Tensor(x_data)), Tensor(y_data)).data

            param.data.flat[i] = orig.flat[i] - epsilon
            loss_minus = loss_fn(model(Tensor(x_data)), Tensor(y_data)).data

            numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)
            param.data.flat[i] = orig.flat[i]

        param.data[:] = orig

        denom = np.maximum(np.abs(analytical[:20]), np.abs(numerical[:20]))
        err = np.max(np.abs(analytical[:20] - numerical[:20]) / np.maximum(denom, 1e-3))
        max_err = max(max_err, err)
        if verbose and err > tol:
            print(f"  [GradientCheck] {name}: relative error = {err:.2e}")

    if verbose:
        print(f"  [GradientCheck] Max relative error: {max_err:.2e}")
    return max_err


def compute_norm(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += np.sum(g ** 2)
    return np.sqrt(total)


def clip_gradients(model, max_norm=1.0):
    params = model.parameters()
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = compute_norm(grads)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for g in grads:
            g *= scale
    return total_norm
