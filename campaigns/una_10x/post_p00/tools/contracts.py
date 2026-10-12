"""Executable DESIGN models; not production caches, locks, IPC or numerical kernels.

Use these pure models as test oracles for invariants in a reviewed implementation.
They deliberately refuse unenumerated scalar types rather than coercing NumPy values.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping


def natural(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def digest(value: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('expected a lower-case SHA-256 digest')
    return value


def _sized(tag: bytes, data: bytes) -> bytes:
    return tag + struct.pack('<Q', len(data)) + data


def typed_frame(value: object) -> bytes:
    """An injective typed framing on this declared small domain, not an engine ABI."""
    if value is None:
        return b'n'
    if type(value) is bool:
        return b'b1' if value else b'b0'
    if type(value) is int:
        return _sized(b'i', str(value).encode('ascii'))
    if type(value) is float:
        return b'f' + struct.pack('<d', value)  # preserves -0 and payload bits
    if type(value) is str:
        return _sized(b's', value.encode('utf-8'))
    if type(value) is bytes:
        return _sized(b'y', value)
    if type(value) in (list, tuple):
        tag = b'l' if type(value) is list else b't'
        return _sized(tag, b''.join(_sized(b'e', typed_frame(x)) for x in value))
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise TypeError('dictionary keys must be exact strings')
        return _sized(b'd', b''.join(typed_frame(k) + typed_frame(value[k])
                                    for k in sorted(value)))
    raise TypeError(f'unsupported exact type: {type(value).__name__}')


@dataclass(frozen=True)
class ArrayBinding:
    """Metadata binding. Actual buffer hashing/reader equivalence are separate duties."""
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    itemsize: int
    nbytes: int
    writable: bool
    alignment_class: str
    content_sha256: str

    def frame(self) -> dict:
        if not self.dtype or not self.alignment_class:
            raise ValueError('dtype and alignment class required')
        if type(self.shape) is not tuple or type(self.strides) is not tuple:
            raise ValueError('shape/strides require tuples')
        if len(self.shape) != len(self.strides):
            raise ValueError('shape/strides rank mismatch')
        count = 1
        for dim in self.shape:
            count *= natural(dim, 'dimension')
        if any(type(s) is not int for s in self.strides):
            raise ValueError('strides require integers')
        if natural(self.itemsize, 'itemsize') == 0:
            raise ValueError('itemsize must be positive')
        if natural(self.nbytes, 'nbytes') != count * self.itemsize:
            raise ValueError('logical payload size mismatch')
        if type(self.writable) is not bool:
            raise ValueError('writable must be boolean')
        digest(self.content_sha256)
        return dict(schema='array-binding-design-1', dtype=self.dtype,
                    shape=self.shape, strides=self.strides, itemsize=self.itemsize,
                    nbytes=self.nbytes, writable=self.writable,
                    alignment_class=self.alignment_class,
                    content_sha256=self.content_sha256)


def metric_key(metric: str, profile: str, trace_id: str, producer: str,
               arrays: Mapping[str, ArrayBinding], scalars: dict,
               implementation: dict, output_cast: str) -> str:
    """Caller supplies the reviewed dependency closure; this model does not infer it."""
    if metric not in {'reach', 'ge', 'gl', 'knn'}:
        raise ValueError('unknown metric')
    if profile not in {'una_legacy', 'madina_legacy', 'corrected_v1'}:
        raise ValueError('unknown profile')
    if not arrays or not implementation or not output_cast:
        raise ValueError('incomplete metric closure')
    frame = dict(schema='metric-id-v2-design', metric=metric, profile=profile,
                 trace=digest(trace_id), producer=digest(producer),
                 arrays={k: v.frame() for k, v in arrays.items()}, scalars=scalars,
                 implementation=implementation, output_cast=output_cast)
    return hashlib.sha256(typed_frame(frame)).hexdigest()


def admit_memory(budget: int, shared: int, workers: int, worker_persistent: int,
                 worker_workspace_peak: int, ready_bytes: int,
                 build_overlap: int, runtime: int, margin: int) -> dict:
    """Byte algebra only; no RSS claim. Workspace argument is an overlapping peak."""
    vals = locals().copy()
    for k, v in vals.items():
        natural(v, k)
    if workers == 0:
        raise ValueError('workers must be positive')
    live = (shared + workers * (worker_persistent + worker_workspace_peak)
            + ready_bytes + build_overlap + runtime + margin)
    return {'admitted': live <= budget, 'modeled_live_bytes': live,
            'budget_bytes': budget, 'headroom_bytes': budget-live,
            'evidence_class': 'hypothetical_sensitivity_analysis'}


class ReadyCredits:
    """Pure reservation model: all dispatched rows have complete-response capacity."""
    def __init__(self, capacity: int):
        self.capacity = natural(capacity, 'capacity')
        self.reservations: dict[int, int] = {}
        self.received: set[int] = set()

    @property
    def used(self) -> int:
        return sum(self.reservations.values())

    def reserve(self, row: int, bound: int) -> bool:
        natural(row, 'row'); natural(bound, 'bound')
        if row in self.reservations or bound == 0:
            raise ValueError('duplicate row or zero response bound')
        if self.used + bound > self.capacity:
            return False
        self.reservations[row] = bound
        return True

    def receive(self, row: int, actual: int) -> None:
        natural(actual, 'actual')
        if row not in self.reservations or row in self.received:
            raise ValueError('unreserved or duplicate response')
        if actual > self.reservations[row]:
            raise ValueError('response exceeded pre-dispatch bound')
        self.received.add(row)

    def release_committed(self, row: int) -> None:
        if row not in self.received:
            raise ValueError('cannot release an unreceived response')
        self.received.remove(row)
        del self.reservations[row]


class CommitModel:
    """Finite publication specification, not filesystem durability implementation.

    dir_synced means the declared directory-sync policy completed. It is not a
    claim that an unsupported filesystem provides stronger durability.
    """
    def __init__(self, file_counts: list[int]):
        if any(type(n) is not int or n < 1 for n in file_counts):
            raise ValueError('each row needs an explicit positive file count')
        self.file_counts = tuple(file_counts)
        self.cursor = 0
        self.verified: set[int] = set()
        self.durable: set[tuple[int, int]] = set()
        self.published: set[tuple[int, int]] = set()
        self.cancelled = False

    def _row(self, row: int) -> None:
        if self.cancelled or type(row) is not int or not 0 <= row < len(self.file_counts):
            raise ValueError('cancelled or invalid row')
        if row < self.cursor:
            raise ValueError('row already committed')

    def _file(self, row: int, file: int) -> None:
        self._row(row)
        if type(file) is not int or not 0 <= file < self.file_counts[row]:
            raise ValueError('invalid file')

    def verify(self, row: int) -> None:
        self._row(row); self.verified.add(row)

    def fsync(self, row: int, file: int) -> None:
        self._file(row, file)
        if row not in self.verified:
            raise ValueError('verify before durability token in this design model')
        self.durable.add((row, file))

    def publish(self, row: int, file: int) -> None:
        self._file(row, file)
        if row != self.cursor or (row, file) not in self.durable:
            raise ValueError('only durable next-row files may publish')
        if (row, file) in self.published:
            raise ValueError('duplicate publication')
        self.published.add((row, file))

    def commit(self, row: int, *, dir_synced: bool, state_applied: bool,
               journaled: bool) -> None:
        self._row(row)
        if (row != self.cursor or not all(x is True for x in
                (dir_synced, state_applied, journaled))
                or any((row, f) not in self.published for f in range(self.file_counts[row]))):
            raise ValueError('incomplete or unordered commit')
        self.cursor += 1

    def cancel(self) -> tuple[int, ...]:
        self.cancelled = True
        return tuple(range(self.cursor))  # partial files are NOT completed rows


def validate_event(event: dict) -> None:
    required = {'run_id', 'row_id', 'pid', 'tid', 'phase', 'start_mono_ns',
                'end_mono_ns', 'status', 'process_cpu_start_ns', 'process_cpu_end_ns'}
    if not required <= event.keys():
        raise ValueError('missing event fields')
    for k in ('run_id', 'phase', 'status'):
        if type(event[k]) is not str or not event[k]:
            raise ValueError(f'nonempty {k} required')
    for k in ('row_id', 'pid', 'tid', 'start_mono_ns', 'end_mono_ns'):
        natural(event[k], k)
    if event['end_mono_ns'] < event['start_mono_ns']:
        raise ValueError('reversed interval')
    cs, ce = event['process_cpu_start_ns'], event['process_cpu_end_ns']
    if (cs is None) != (ce is None):
        raise ValueError('CPU endpoints both available or both null')
    if cs is not None:
        natural(cs, 'CPU start'); natural(ce, 'CPU end')
        if ce < cs:
            raise ValueError('CPU counter reversed')


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    """Diagnostic union, not a stage sum or a causally identified critical path."""
    spans = []
    for a, b in intervals:
        natural(a, 'start'); natural(b, 'end')
        if b < a:
            raise ValueError('reversed interval')
        spans.append((a, b))
    total = 0
    end = -1
    for a, b in sorted(spans):
        if b > end:
            total += b - max(a, end)
            end = b
    return total
