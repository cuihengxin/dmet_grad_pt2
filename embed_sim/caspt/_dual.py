"""First-order real dual arrays for the in-core CASPT2 reference gradient.

This deliberately small implementation differentiates algebraic operations,
not coordinate-displaced energies. It is not a general NumPy replacement.
"""
import numpy as np


class Dual:
    __array_priority__ = 10000

    def __init__(self, value, tangent=None):
        self.v = np.asarray(value)
        self.d = np.zeros_like(self.v) if tangent is None else np.asarray(tangent)

    @property
    def shape(self): return self.v.shape
    @property
    def ndim(self): return self.v.ndim
    @property
    def size(self): return self.v.size
    @property
    def dtype(self): return self.v.dtype
    @property
    def T(self): return self.transpose()
    def __len__(self): return len(self.v)
    def __getitem__(self, key): return Dual(self.v[key], self.d[key])
    def __setitem__(self, key, value):
        value = dual(value)
        self.v[key], self.d[key] = value.v, value.d
    def copy(self): return Dual(self.v.copy(), self.d.copy())
    def conj(self): return Dual(self.v.conj(), self.d.conj())
    def reshape(self, *shape): return Dual(self.v.reshape(*shape), self.d.reshape(*shape))
    def ravel(self): return self.reshape(-1)
    def transpose(self, *axes):
        return Dual(self.v.transpose(*axes), self.d.transpose(*axes))
    def __neg__(self): return Dual(-self.v, -self.d)
    def __add__(self, other):
        other = dual(other)
        return Dual(self.v + other.v, self.d + other.d)
    __radd__ = __add__
    def __sub__(self, other): return self + -dual(other)
    def __rsub__(self, other): return dual(other) + -self
    def __mul__(self, other):
        other = dual(other)
        return Dual(self.v * other.v, self.d * other.v + self.v * other.d)
    __rmul__ = __mul__
    def __truediv__(self, other):
        other = dual(other)
        return Dual(self.v / other.v, (self.d - self.v / other.v * other.d) / other.v)
    def __rtruediv__(self, other): return dual(other) / self
    def __pow__(self, power):
        return Dual(self.v ** power, power * self.v ** (power - 1) * self.d)
    def __matmul__(self, other):
        other = dual(other)
        return Dual(self.v @ other.v, self.d @ other.v + self.v @ other.d)
    def __rmatmul__(self, other): return dual(other) @ self
    def __gt__(self, other): return self.v > value(other)
    def __lt__(self, other): return self.v < value(other)
    def __abs__(self): return np.abs(self.v)


def dual(x): return x if isinstance(x, Dual) else Dual(x)
def value(x): return x.v if isinstance(x, Dual) else x


def multilinear(func, *args, **kwargs):
    vals = [value(x) for x in args]
    out = func(*vals, **kwargs)
    deriv = np.zeros_like(out)
    for i, arg in enumerate(args):
        if isinstance(arg, Dual):
            xs = vals.copy()
            xs[i] = arg.d
            deriv += func(*xs, **kwargs)
    return Dual(out, deriv)


def einsum(backend, script, *args, alpha=1., beta=0., out=None):
    result = alpha * multilinear(lambda *xs: np.einsum(script, *xs, optimize=True), *args)
    if out is not None:
        result = result + beta * out
        out[...] = result
        return out
    return result


def eigh(mat):
    mat = dual(mat)
    w, u = np.linalg.eigh(mat.v)
    # Degenerate perturbation theory: choose a directional eigenbasis within
    # each degenerate space before differentiating the eigenvectors.
    start = 0
    while start < len(w):
        end = start + 1
        while end < len(w) and abs(w[end] - w[start]) < 1e-10:
            end += 1
        if end - start > 1:
            block = u[:, start:end]
            _, rotation = np.linalg.eigh(block.T @ mat.d @ block)
            u[:, start:end] = block @ rotation
        start = end
    rotated = u.T @ mat.d @ u
    gap = w[None, :] - w[:, None]
    inv = np.zeros_like(gap)
    mask = np.abs(gap) > 1e-10
    inv[mask] = 1 / gap[mask]
    # A gauge within degenerate spaces is immaterial if all members are kept.
    return Dual(w, np.diag(rotated)), Dual(u, u @ (rotated * inv))


class Numpy:
    ndarray = (np.ndarray, Dual)

    def __getattr__(self, name):
        if name in ('abs', 'max', 'array', 'identity', 'ogrid', 'cumsum',
                    'tril_indices', 'triu_indices'):
            return getattr(np, name)
        raise NotImplementedError('Dual NumPy operation: ' + name)
    def zeros(self, shape, dtype=float): return Dual(np.zeros(shape, dtype=dtype))
    empty = zeros
    def zeros_like(self, x): return Dual(np.zeros_like(value(x)))
    def dot(self, a, b): return multilinear(np.dot, a, b)
    def diag(self, a):
        a = dual(a)
        return Dual(np.diag(a.v), np.diag(a.d))
    def concatenate(self, arrays, axis=0):
        arrays = [dual(a) for a in arrays]
        return Dual(np.concatenate([a.v for a in arrays], axis),
                    np.concatenate([a.d for a in arrays], axis))
    def split(self, a, indices):
        a = dual(a)
        return [Dual(v, d) for v, d in zip(np.split(a.v, indices), np.split(a.d, indices))]
    def tensordot(self, a, b, axes): return multilinear(np.tensordot, a, b, axes=axes)
