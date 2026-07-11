import numpy as np

def xavier(shape, gain=1.0):
    if len(shape) >= 2:
        fan_in = shape[1] if len(shape) > 1 else 1
        fan_out = shape[0]
        limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape).astype(np.float32)
    return np.random.randn(*shape).astype(np.float32) * 0.01

def xavier_normal(shape, gain=1.0):
    if len(shape) >= 2:
        fan_in = shape[1] if len(shape) > 1 else 1
        fan_out = shape[0]
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape).astype(np.float32) * std
    return np.random.randn(*shape).astype(np.float32) * 0.01

def he(shape, negative_slope=0.0):
    if len(shape) >= 2:
        fan_in = shape[1] if len(shape) > 1 else 1
        std = np.sqrt(2.0 / (fan_in * (1 + negative_slope**2)))
        return np.random.randn(*shape).astype(np.float32) * std
    return np.random.randn(*shape).astype(np.float32) * 0.01

def he_normal(shape, negative_slope=0.0):
    return he(shape, negative_slope)

def orthogonal(shape, gain=1.0):
    if len(shape) < 2:
        return np.random.randn(*shape).astype(np.float32) * 0.01
    flat_shape = (shape[0], np.prod(shape[1:]).item())
    a = np.random.randn(*flat_shape).astype(np.float32)
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else vh
    q = q.reshape(shape).astype(np.float32)
    return q * gain

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def ones(shape):
    return np.ones(shape, dtype=np.float32)

def kaiming(shape, mode='fan_in', nonlinearity='relu'):
    if len(shape) >= 2:
        fan = shape[1] if mode == 'fan_in' else shape[0]
        gain = { 'relu': np.sqrt(2.0), 'leaky_relu': np.sqrt(2.0/1.0),
                 'tanh': 1.0, 'sigmoid': 1.0, 'linear': 1.0 }.get(nonlinearity, 1.0)
        std = gain / np.sqrt(fan)
        return np.random.randn(*shape).astype(np.float32) * std
    return np.random.randn(*shape).astype(np.float32) * 0.01
