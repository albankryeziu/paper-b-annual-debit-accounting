from __future__ import annotations


# ============================================================================
# final_runtime.active_compression (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Exact active-coordinate and inactive-row compression for Gurobi matrix models.

The implementation removes only coordinates whose declared lower and upper
bounds are both exactly zero.  Therefore the reduced model is an exact
reparameterization x = P x_active, not an approximation or a changed model.
Original-coordinate values are reconstructed before reporting and independent
replay.  Matrix constraints additionally omit only all-zero rows that are
tautologies; infeasible zero rows are retained.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence
import math
import weakref

import numpy as np
try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is a declared runtime dependency
    sp = None


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, tuple):
        out = tuple(int(v) for v in shape)
    elif isinstance(shape, list):
        out = tuple(int(v) for v in shape)
    elif np.isscalar(shape):
        out = (int(shape),)
    else:
        arr = tuple(int(v) for v in shape)
        out = arr
    if not out or any(v < 0 for v in out):
        raise ValueError(f"invalid MVar shape: {shape!r}")
    return out


def _broadcast_flat(value: Any, shape: tuple[int, ...], *, default: float) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64))
    if value is None:
        return np.full(size, default, dtype=float)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(size, arr.item(), dtype=arr.dtype if arr.dtype != object else float)
    try:
        return np.broadcast_to(arr, shape).reshape(-1).copy()
    except ValueError:
        if arr.size != size:
            raise
        return arr.reshape(-1).copy()


def _take_vector(value: Any, active: np.ndarray, shape: tuple[int, ...]) -> Any:
    if value is None or np.isscalar(value) or isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return value
    size = int(np.prod(shape, dtype=np.int64))
    if arr.size != size:
        try:
            arr = np.broadcast_to(arr, shape)
        except ValueError:
            return value
    return np.asarray(arr).reshape(-1)[active]


def _is_sparse(value: Any) -> bool:
    return sp is not None and sp.issparse(value)


def _select_rows(matrix: Any, rows: np.ndarray) -> Any:
    if _is_sparse(matrix):
        return matrix.tocsr()[rows, :]
    return np.asarray(matrix)[rows, ...]


def _select_columns(matrix: Any, columns: np.ndarray) -> Any:
    if _is_sparse(matrix):
        return matrix.tocsc()[:, columns]
    return np.asarray(matrix)[..., columns]


def _matrix_shape(matrix: Any) -> tuple[int, ...]:
    shape = getattr(matrix, 'shape', None)
    return tuple(shape) if shape is not None else tuple(np.asarray(matrix).shape)


@dataclass(frozen=True)
class ActiveCoordinateMap:
    """Map between the frozen scientific coordinate system and active variables."""

    full_shape: tuple[int, ...]
    active_flat: np.ndarray

    @classmethod
    def from_bounds(cls, shape: Any, lb: Any, ub: Any) -> 'ActiveCoordinateMap':
        full_shape = _shape_tuple(shape)
        lower = _broadcast_flat(lb, full_shape, default=0.0)
        upper = _broadcast_flat(ub, full_shape, default=np.inf)
        # Exact, fail-safe rule: compress only coordinates fixed exactly at zero.
        fixed_zero = np.equal(lower, 0) & np.equal(upper, 0)
        active = np.flatnonzero(~fixed_zero).astype(np.int64, copy=False)
        return cls(full_shape=full_shape, active_flat=active)

    @property
    def full_size(self) -> int:
        return int(np.prod(self.full_shape, dtype=np.int64))

    @property
    def active_size(self) -> int:
        return int(self.active_flat.size)

    @property
    def dropped_size(self) -> int:
        return self.full_size - self.active_size

    @property
    def is_identity(self) -> bool:
        return self.active_size == self.full_size

    def compress_vector(self, value: Any) -> Any:
        return _take_vector(value, self.active_flat, self.full_shape)

    def compress_columns(self, matrix: Any) -> Any:
        shape = _matrix_shape(matrix)
        if not shape or shape[-1] != self.full_size:
            raise ValueError(
                f"matrix has {shape[-1] if shape else 'no'} columns; "
                f"expected {self.full_size} original coordinates"
            )
        return _select_columns(matrix, self.active_flat)

    def compress_rows(self, matrix: Any) -> Any:
        shape = _matrix_shape(matrix)
        if not shape or shape[0] != self.full_size:
            raise ValueError(
                f"matrix has {shape[0] if shape else 'no'} rows; "
                f"expected {self.full_size} original coordinates"
            )
        return _select_rows(matrix, self.active_flat)

    def expand(self, active_value: Any, *, fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(active_value)
        if arr.size != self.active_size:
            raise ValueError(f"expected {self.active_size} active values, got {arr.size}")
        full = np.full(self.full_size, fill, dtype=arr.dtype if arr.dtype != object else float)
        full[self.active_flat] = arr.reshape(-1)
        return full.reshape(self.full_shape)

    def selector(self) -> Any:
        if sp is None:
            dense = np.zeros((self.full_size, self.active_size), dtype=float)
            dense[self.active_flat, np.arange(self.active_size)] = 1.0
            return dense
        data = np.ones(self.active_size, dtype=float)
        rows = self.active_flat
        cols = np.arange(self.active_size, dtype=np.int64)
        return sp.csc_matrix((data, (rows, cols)), shape=(self.full_size, self.active_size))


@dataclass(frozen=True)
class ConstraintRowMap:
    full_size: int
    kept_rows: np.ndarray
    dropped_rows: np.ndarray
    rhs: np.ndarray
    sense: np.ndarray

    def expand(self, values: Any, *, dropped_fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(values).reshape(-1)
        if arr.size != self.kept_rows.size:
            raise ValueError('constraint result has unexpected size')
        out = np.full(self.full_size, dropped_fill, dtype=arr.dtype if arr.dtype != object else float)
        out[self.kept_rows] = arr
        return out


_STATS: dict[int, dict[str, int]] = {}
_MODEL_REFS: dict[int, weakref.ReferenceType[Any]] = {}


def _stats(model: Any) -> dict[str, int]:
    key = id(model)
    if key not in _STATS:
        _STATS[key] = {
            'mvar_blocks_seen': 0,
            'mvar_blocks_compressed': 0,
            'coordinates_declared': 0,
            'coordinates_active': 0,
            'coordinates_dropped_fixed_zero': 0,
            'matrix_rows_seen': 0,
            'matrix_rows_kept': 0,
            'matrix_rows_dropped_tautological': 0,
        }
        try:
            _MODEL_REFS[key] = weakref.ref(model, lambda _ref, k=key: (_STATS.pop(k, None), _MODEL_REFS.pop(k, None)))
        except TypeError:
            pass
    return _STATS[key]


def compression_stats(model: Any) -> dict[str, int]:
    return dict(_stats(model))


def _normalise_addmvar_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if args:
        shape = args[0]
        remaining = list(args[1:])
    elif 'shape' in kwargs:
        shape = kwargs.pop('shape')
        remaining = []
    else:
        raise TypeError('addMVar requires shape')
    names = ('lb', 'ub', 'obj', 'vtype', 'name')
    if len(remaining) > len(names):
        raise TypeError('too many positional arguments for addMVar')
    for key, value in zip(names, remaining):
        if key in kwargs:
            raise TypeError(f'multiple values for {key}')
        kwargs[key] = value
    return shape, kwargs


class ActiveMVar:
    """Virtual full-coordinate MVar backed by only non-fixed coordinates."""

    __array_priority__ = 100000

    def __init__(self, native: Any, mapping: ActiveCoordinateMap, lb: Any, ub: Any, model: Any):
        self.native = native
        self.mapping = mapping
        self._lb = _broadcast_flat(lb, mapping.full_shape, default=0.0).reshape(mapping.full_shape)
        self._ub = _broadcast_flat(ub, mapping.full_shape, default=np.inf).reshape(mapping.full_shape)
        self._model = model
        self._full_expr_cache: Any = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mapping.full_shape

    @property
    def size(self) -> int:
        return self.mapping.full_size

    @property
    def ndim(self) -> int:
        return len(self.mapping.full_shape)

    @property
    def T(self) -> Any:
        return self._full_expr().T

    @property
    def X(self) -> np.ndarray:
        return self.mapping.expand(self.native.X)

    @property
    def Xn(self) -> np.ndarray:
        return self.mapping.expand(self.native.Xn)

    @property
    def Start(self) -> np.ndarray:
        return self.mapping.expand(self.native.Start, fill=np.nan)

    @Start.setter
    def Start(self, value: Any) -> None:
        self.native.Start = self.mapping.compress_vector(value)

    @property
    def LB(self) -> np.ndarray:
        return np.array(self._lb, copy=True)

    @LB.setter
    def LB(self, value: Any) -> None:
        arr = _broadcast_flat(value, self.shape, default=0.0).reshape(self.shape)
        # Changing a removed coordinate away from zero would invalidate exactness.
        dropped = np.ones(self.size, dtype=bool)
        dropped[self.mapping.active_flat] = False
        if np.any(arr.reshape(-1)[dropped] != 0):
            raise ValueError('cannot unfix a compressed zero coordinate; rebuild the model')
        self._lb = arr
        self.native.LB = self.mapping.compress_vector(arr)

    @property
    def UB(self) -> np.ndarray:
        return np.array(self._ub, copy=True)

    @UB.setter
    def UB(self, value: Any) -> None:
        arr = _broadcast_flat(value, self.shape, default=np.inf).reshape(self.shape)
        dropped = np.ones(self.size, dtype=bool)
        dropped[self.mapping.active_flat] = False
        if np.any(arr.reshape(-1)[dropped] != 0):
            raise ValueError('cannot unfix a compressed zero coordinate; rebuild the model')
        self._ub = arr
        self.native.UB = self.mapping.compress_vector(arr)

    def _full_expr(self) -> Any:
        if self._full_expr_cache is None:
            expr = self.mapping.selector() @ self.native
            if self.shape != (self.size,):
                expr = expr.reshape(self.shape)
            self._full_expr_cache = expr
        return self._full_expr_cache

    def full_expression(self) -> Any:
        return self._full_expr()

    def reshape(self, *shape: Any, order: str = 'C') -> Any:
        return self._full_expr().reshape(*shape, order=order)

    def sum(self, axis: Any = None) -> Any:
        return self._full_expr().sum(axis=axis)

    def item(self) -> Any:
        return self._full_expr().item()

    def tolist(self) -> Any:
        return self._full_expr().tolist()

    def getAttr(self, attr: str) -> Any:
        if attr in {'X', 'Xn', 'Start', 'LB', 'UB'}:
            return getattr(self, attr)
        value = self.native.getAttr(attr)
        arr = np.asarray(value)
        return self.mapping.expand(arr) if arr.size == self.mapping.active_size else value

    def setAttr(self, attr: str, value: Any) -> None:
        if attr in {'Start', 'LB', 'UB'}:
            setattr(self, attr, value)
            return
        self.native.setAttr(attr, self.mapping.compress_vector(value))

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[Any]:
        for i in range(self.shape[0]):
            yield self[i]

    def __getitem__(self, key: Any) -> Any:
        full_indices = np.arange(self.size, dtype=np.int64).reshape(self.shape)
        selected = full_indices[key]
        if np.isscalar(selected):
            flat = int(selected)
            pos = int(np.searchsorted(self.mapping.active_flat, flat))
            if pos < self.mapping.active_size and int(self.mapping.active_flat[pos]) == flat:
                return self.native[pos]
            return 0.0
        selected_arr = np.asarray(selected)
        flat = selected_arr.reshape(-1)
        inverse = {int(v): i for i, v in enumerate(self.mapping.active_flat.tolist())}
        rows, cols = [], []
        for row, original in enumerate(flat.tolist()):
            col = inverse.get(int(original))
            if col is not None:
                rows.append(row)
                cols.append(col)
        if sp is None:
            selector = np.zeros((flat.size, self.mapping.active_size), dtype=float)
            if rows:
                selector[np.asarray(rows), np.asarray(cols)] = 1.0
        else:
            selector = sp.csr_matrix(
                (np.ones(len(rows)), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
                shape=(flat.size, self.mapping.active_size),
            )
        expr = selector @ self.native
        return expr.reshape(selected_arr.shape) if selected_arr.shape else expr.item()

    def __neg__(self) -> Any: return -self._full_expr()
    def __pos__(self) -> Any: return self._full_expr()
    def __add__(self, other: Any) -> Any: return self._full_expr() + unwrap_active(other)
    def __radd__(self, other: Any) -> Any: return unwrap_active(other) + self._full_expr()
    def __sub__(self, other: Any) -> Any: return self._full_expr() - unwrap_active(other)
    def __rsub__(self, other: Any) -> Any: return unwrap_active(other) - self._full_expr()
    def __mul__(self, other: Any) -> Any: return self._full_expr() * unwrap_active(other)
    def __rmul__(self, other: Any) -> Any: return unwrap_active(other) * self._full_expr()
    def __truediv__(self, other: Any) -> Any: return self._full_expr() / unwrap_active(other)
    def __pow__(self, other: Any) -> Any: return self._full_expr() ** other
    def __le__(self, other: Any) -> Any: return self._full_expr() <= unwrap_active(other)
    def __ge__(self, other: Any) -> Any: return self._full_expr() >= unwrap_active(other)
    def __eq__(self, other: Any) -> Any: return self._full_expr() == unwrap_active(other)

    def __rmatmul__(self, left: Any) -> Any:
        try:
            shape = _matrix_shape(left)
            if self.ndim == 1 and shape and shape[-1] == self.size:
                return self.mapping.compress_columns(left) @ self.native
        except Exception:
            pass
        return left @ self._full_expr()

    def __matmul__(self, right: Any) -> Any:
        try:
            shape = _matrix_shape(right)
            if self.ndim == 1 and shape and shape[0] == self.size:
                return self.native @ self.mapping.compress_rows(right)
        except Exception:
            pass
        return self._full_expr() @ right

    def __getattr__(self, name: str) -> Any:
        # Native scalar metadata remains available. Methods that depend on the
        # original shape are implemented explicitly above.
        return getattr(self.native, name)


class ActiveMConstr:
    def __init__(self, native: Any, mapping: ConstraintRowMap):
        self.native = native
        self.mapping = mapping

    @property
    def Pi(self) -> np.ndarray:
        return self.mapping.expand(self.native.Pi, dropped_fill=0.0)

    @property
    def Slack(self) -> np.ndarray:
        return self.mapping.expand(self.native.Slack, dropped_fill=0.0)

    @property
    def IISConstr(self) -> np.ndarray:
        return self.mapping.expand(self.native.IISConstr, dropped_fill=0.0)

    def getAttr(self, attr: str) -> Any:
        value = self.native.getAttr(attr)
        arr = np.asarray(value)
        if arr.size == self.mapping.kept_rows.size:
            return self.mapping.expand(arr, dropped_fill=0.0)
        return value

    def setAttr(self, attr: str, value: Any) -> None:
        arr = np.asarray(value)
        if arr.size == self.mapping.full_size:
            value = arr.reshape(-1)[self.mapping.kept_rows]
        self.native.setAttr(attr, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)


def unwrap_active(value: Any) -> Any:
    if isinstance(value, ActiveMVar):
        return value.full_expression()
    return value


def add_active_mvar(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Drop exactly fixed-zero coordinates before calling ``Model.addMVar``."""
    shape, params = _normalise_addmvar_args(args, dict(kwargs))
    lb = params.get('lb', 0.0)
    ub = params.get('ub', np.inf)
    mapping = ActiveCoordinateMap.from_bounds(shape, lb, ub)
    stats = _stats(model)
    stats['mvar_blocks_seen'] += 1
    stats['coordinates_declared'] += mapping.full_size
    stats['coordinates_active'] += mapping.active_size
    stats['coordinates_dropped_fixed_zero'] += mapping.dropped_size

    # Avoid a zero-dimensional Gurobi object.  This fallback does not alter the
    # model and occurs only when a whole declared block is fixed to zero.
    if mapping.is_identity or mapping.active_size == 0:
        return model.addMVar(shape, **params)

    reduced = dict(params)
    for key in ('lb', 'ub', 'obj', 'vtype'):
        if key in reduced:
            reduced[key] = mapping.compress_vector(reduced[key])
    native = model.addMVar(mapping.active_size, **reduced)
    stats['mvar_blocks_compressed'] += 1
    return ActiveMVar(native=native, mapping=mapping, lb=lb, ub=ub, model=model)


def _sense_array(sense: Any, rows: int) -> np.ndarray:
    if isinstance(sense, str) and len(sense) == 1:
        return np.full(rows, sense, dtype='<U1')
    arr = np.asarray(sense)
    if arr.ndim == 0:
        return np.full(rows, str(arr.item()), dtype='<U1')
    return np.broadcast_to(arr, (rows,)).astype('<U1', copy=True)


def _rhs_array(rhs: Any, rows: int) -> np.ndarray:
    arr = np.asarray(rhs)
    if arr.ndim == 0:
        return np.full(rows, float(arr.item()), dtype=float)
    return np.broadcast_to(arr, (rows,)).astype(float, copy=True)


def _row_nnz(matrix: Any) -> np.ndarray:
    if _is_sparse(matrix):
        return np.diff(matrix.tocsr().indptr)
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError('matrix constraint coefficient array must be two-dimensional')
    return np.count_nonzero(arr, axis=1)


def _tautological_zero_rows(matrix: Any, sense: Any, rhs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = int(_matrix_shape(matrix)[0])
    senses = _sense_array(sense, rows)
    right = _rhs_array(rhs, rows)
    zero = _row_nnz(matrix) == 0
    tautology = zero & (
        ((senses == '<') & (right >= 0))
        | ((senses == '>') & (right <= 0))
        | ((senses == '=') & (right == 0))
    )
    kept = np.flatnonzero(~tautology).astype(np.int64, copy=False)
    dropped = np.flatnonzero(tautology).astype(np.int64, copy=False)
    return kept, dropped, senses, right


def add_mconstr_active(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Compress columns through x=P x_active and close only nontrivial row keys."""
    if len(args) >= 4:
        A, x, sense, b, *tail = args
        name = tail[0] if tail else kwargs.pop('name', '')
    else:
        local = dict(kwargs)
        try:
            A = args[0] if len(args) > 0 else local.pop('A')
            x = args[1] if len(args) > 1 else local.pop('x')
            sense = args[2] if len(args) > 2 else local.pop('sense')
            b = local.pop('b')
        except KeyError as exc:
            raise TypeError('addMConstr requires A, x, sense, b') from exc
        name = local.pop('name', '')
        if local:
            raise TypeError(f'unexpected addMConstr arguments: {sorted(local)}')

    if isinstance(x, ActiveMVar):
        A = x.mapping.compress_columns(A)
        x_native = x.native
    else:
        x_native = x

    kept, dropped, senses, right = _tautological_zero_rows(A, sense, b)
    rows = int(_matrix_shape(A)[0])
    stats = _stats(model)
    stats['matrix_rows_seen'] += rows
    stats['matrix_rows_kept'] += int(kept.size)
    stats['matrix_rows_dropped_tautological'] += int(dropped.size)
    if dropped.size == 0:
        return model.addMConstr(A, x_native, sense, b, name=name)

    reduced_A = _select_rows(A, kept)
    reduced_sense = senses[kept]
    reduced_b = right[kept]
    native = model.addMConstr(reduced_A, x_native, reduced_sense, reduced_b, name=name)
    mapping = ConstraintRowMap(rows, kept, dropped, right, senses)
    return ActiveMConstr(native, mapping)


def set_mobjective_active(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply exact selector projections to matrix objectives."""
    names = ('Q', 'c', 'constant', 'xQ_L', 'xQ_R', 'xc', 'sense')
    values = dict(kwargs)
    for key, value in zip(names, args):
        if key in values:
            raise TypeError(f'multiple values for {key}')
        values[key] = value
    if not all(k in values for k in ('Q', 'c', 'constant')):
        raise TypeError('setMObjective requires Q, c, constant')
    Q = values.get('Q')
    c = values.get('c')
    constant = values.get('constant')
    x_left = values.get('xQ_L')
    x_right = values.get('xQ_R')
    x_linear = values.get('xc')
    sense = values.get('sense')

    if isinstance(x_left, ActiveMVar):
        Q = x_left.mapping.compress_rows(Q)
        x_left = x_left.native
        if x_right is None:
            # Gurobi's omitted right block denotes the same quadratic block.
            Q = _select_columns(Q, values['xQ_L'].mapping.active_flat)
            x_right = x_left
    if isinstance(x_right, ActiveMVar):
        Q = x_right.mapping.compress_columns(Q)
        x_right = x_right.native
    if isinstance(x_linear, ActiveMVar):
        c = x_linear.mapping.compress_vector(c)
        x_linear = x_linear.native

    call = {'xQ_L': x_left, 'xQ_R': x_right, 'xc': x_linear}
    if sense is not None:
        call['sense'] = sense
    return model.setMObjective(Q, c, constant, **call)


def active_hstack(values: Sequence[Any]) -> Any:
    import gurobipy as gp
    return gp.hstack([unwrap_active(v) for v in values])


def active_vstack(values: Sequence[Any]) -> Any:
    import gurobipy as gp
    return gp.vstack([unwrap_active(v) for v in values])


def _contains_active(values: Any) -> bool:
    if isinstance(values, ActiveMVar):
        return True
    if isinstance(values, (list, tuple)):
        return any(_contains_active(v) for v in values)
    return False


def active_np_concatenate(values: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
    if not _contains_active(values):
        return np.concatenate(values, *args, **kwargs)
    axis = kwargs.get('axis', args[0] if args else 0)
    if axis not in (None, 0, -1):
        raise ValueError('Gurobi expression concatenation supports vector axis only')
    return active_hstack([unwrap_active(v).reshape(-1) if hasattr(unwrap_active(v), 'reshape') else v for v in values])


def active_np_hstack(values: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
    if not _contains_active(values):
        return np.hstack(values, *args, **kwargs)
    return active_hstack(values)


def active_np_vstack(values: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
    if not _contains_active(values):
        return np.vstack(values, *args, **kwargs)
    return active_vstack(values)
