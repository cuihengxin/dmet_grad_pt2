"""Reverse tensor contractions for the real IC-CASPT2 Lagrangian.

Nodes are immutable; assignment creates a new version. Only operations used by
CASPT2 are implemented, and unsupported NumPy operations fail explicitly.
"""
import numpy as np


class _Node:
    def __init__(self, value, parents=(), requires=False):
        self.value = np.asarray(value)
        self.parents = tuple((node, fn) for node, fn in parents if node.requires)
        self.requires = requires or bool(self.parents)
        self.bar = None


def _unbroadcast(bar, shape):
    while bar.ndim > len(shape):
        bar = bar.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and bar.shape[axis] != 1:
            bar = bar.sum(axis=axis, keepdims=True)
    return bar.reshape(shape)


def tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def value(x):
    return x.v if isinstance(x, Tensor) else x


def _result(value, parents):
    out = Tensor.__new__(Tensor)
    out.node = _Node(value, parents)
    return out


class Tensor:
    __array_priority__ = 10000

    def __init__(self, value, requires_grad=False):
        self.node = _Node(value, requires=requires_grad)

    @property
    def v(self): return self.node.value
    @property
    def grad(self):
        return np.zeros_like(self.v) if self.node.bar is None else self.node.bar
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
    def __abs__(self): return np.abs(self.v)
    def __gt__(self, other): return self.v > value(other)
    def __lt__(self, other): return self.v < value(other)

    def __getitem__(self, key):
        node = self.node
        def back(bar):
            out = np.zeros_like(node.value)
            np.add.at(out, key, bar)
            return out
        return _result(self.v[key], [(node, back)])

    def __setitem__(self, key, x):
        old, x = self.node, tensor(x).node
        out = old.value.copy()
        out[key] = x.value
        def old_back(bar):
            result = bar.copy()
            result[key] = 0
            return result
        self.node = _Node(out, [(old, old_back),
                               (x, lambda bar: _unbroadcast(bar[key], x.value.shape))])

    def copy(self): return _result(self.v.copy(), [(self.node, lambda b: b)])
    def conj(self): return self.copy()
    def ravel(self): return self.reshape(-1)
    def reshape(self, *shape):
        old = self.node
        return _result(self.v.reshape(*shape), [(old, lambda b: b.reshape(old.value.shape))])
    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        inverse = np.argsort(axes)
        return _result(self.v.transpose(axes), [(self.node, lambda b: b.transpose(inverse))])
    def __neg__(self): return _result(-self.v, [(self.node, lambda b: -b)])
    def __add__(self, other):
        a, b = self.node, tensor(other).node
        return _result(a.value+b.value,
            [(a, lambda bar: _unbroadcast(bar, a.value.shape)),
             (b, lambda bar: _unbroadcast(bar, b.value.shape))])
    __radd__ = __add__
    def __sub__(self, other): return self + -tensor(other)
    def __rsub__(self, other): return tensor(other) + -self
    def __mul__(self, other):
        a, b = self.node, tensor(other).node
        return _result(a.value*b.value,
            [(a, lambda bar: _unbroadcast(bar*b.value, a.value.shape)),
             (b, lambda bar: _unbroadcast(bar*a.value, b.value.shape))])
    __rmul__ = __mul__
    def __truediv__(self, other):
        a, b = self.node, tensor(other).node
        return _result(a.value/b.value,
            [(a, lambda bar: _unbroadcast(bar/b.value, a.value.shape)),
             (b, lambda bar: _unbroadcast(-bar*a.value/b.value**2, b.value.shape))])
    def __rtruediv__(self, other): return tensor(other) / self
    def __pow__(self, power):
        a = self.node
        return _result(a.value**power, [(a, lambda b: b*power*a.value**(power-1))])
    def __matmul__(self, other):
        # All matrix products in the tensor equations are rank two.
        return einsum(None, 'ij,jk->ik', self, other)
    def __rmatmul__(self, other): return tensor(other) @ self

    def backward(self):
        if self.size != 1:
            raise ValueError('Reverse contraction requires a scalar output')
        order, seen, stack = [], set(), [(self.node, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p, _ in node.parents)
        for node in order:
            node.bar = None
        self.node.bar = np.ones_like(self.v)
        for node in reversed(order):
            if node.bar is None:
                continue
            for parent, fn in node.parents:
                update = fn(node.bar)
                parent.bar = update if parent.bar is None else parent.bar + update
            if node.parents:
                node.bar = None


def _einsum_back(script, operands, target, bar):
    lhs, out = script.split('->')
    labels = lhs.split(',')
    wanted = labels[target]
    unique = ''.join(dict.fromkeys(wanted))
    terms, arrays = [out], [bar]
    for i, (label, operand) in enumerate(zip(labels, operands)):
        if i != target:
            terms.append(label)
            arrays.append(operand)
    present = ''.join(terms)
    for label in unique:
        if label not in present:
            terms.append(label)
            arrays.append(np.ones(operands[target].shape[wanted.index(label)]))
    result = np.einsum(','.join(terms)+'->'+unique, *arrays, optimize=True)
    if len(unique) == len(wanted):
        return result
    # Repeated input indices are diagonal gathers; their adjoint scatters.
    full = np.zeros_like(operands[target])
    grid = np.ix_(*[np.arange(operands[target].shape[wanted.index(s)]) for s in unique])
    full[tuple(grid[unique.index(s)] for s in wanted)] = result
    return full


def einsum(backend, script, *args, alpha=1., beta=0., out=None):
    nodes = [tensor(x).node for x in args]
    operands = [node.value for node in nodes]
    parents = [(node, lambda bar, i=i: _einsum_back(script, operands, i, bar))
               for i, node in enumerate(nodes)]
    result = alpha * _result(np.einsum(script, *operands, optimize=True), parents)
    if out is not None:
        result = result + beta * out
        out[...] = result
        return out
    return result


def eigh(mat):
    mat = tensor(mat)
    w, u = np.linalg.eigh(mat.v)
    gap = w[None, :] - w[:, None]
    inverse = np.zeros_like(gap)
    mask = np.abs(gap) > 1e-10
    inverse[mask] = 1 / gap[mask]
    def w_back(bar): return (u * bar) @ u.T
    def u_back(bar):
        mid = inverse * (u.T @ bar)
        return u @ ((mid + mid.T) / 2) @ u.T
    return _result(w, [(mat.node, w_back)]), _result(u, [(mat.node, u_back)])


class Numpy:
    ndarray = (np.ndarray, Tensor)
    def __getattr__(self, name):
        if name in ('abs', 'max', 'array', 'identity', 'ogrid', 'cumsum',
                    'tril_indices', 'triu_indices'):
            return getattr(np, name)
        raise NotImplementedError('Reverse NumPy operation: '+name)
    def zeros(self, shape, dtype=float): return Tensor(np.zeros(shape, dtype=dtype))
    empty = zeros
    def dot(self, a, b):
        if tensor(a).ndim != 1 or tensor(b).ndim != 1:
            raise NotImplementedError('Reverse dot supports vectors')
        return einsum(None, 'i,i->', a, b)
    def diag(self, a):
        a = tensor(a)
        if a.ndim == 2:
            return _result(np.diag(a.v), [(a.node, np.diag)])
        return _result(np.diag(a.v), [(a.node, lambda b: np.diag(b))])
    def concatenate(self, arrays, axis=0):
        arrays = [tensor(a) for a in arrays]
        starts = np.cumsum([0]+[a.shape[axis] for a in arrays])
        parents = []
        for i, a in enumerate(arrays):
            key = [slice(None)] * a.ndim
            key[axis] = slice(starts[i], starts[i+1])
            parents.append((a.node, lambda bar, key=tuple(key): bar[key]))
        return _result(np.concatenate([a.v for a in arrays], axis), parents)
    def split(self, a, indices):
        a = tensor(a)
        boundaries = [0]+list(indices)+[len(a)]
        return [a[i:j] for i, j in zip(boundaries[:-1], boundaries[1:])]
