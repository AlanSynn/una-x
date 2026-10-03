"""Reference canonical encoder for the una_10x identity spec
(schema_version una-10x-c10-1).

Frame law:
- fields are framed in SCHEMA ORDER (explicit pair list, never sorted —
  reordering fields is an identity change by construction);
- str   -> b"s" + u64len + utf8
- int   -> b"i" + 8-byte little-endian signed
- float -> b"f" + 8-byte IEEE-754 bit pattern, little-endian: BIT identity,
  so -0.0 != 0.0 and every 1-ULP change is visible;
- bytes -> b"b" + u64len + raw
- bool  -> b"B" + 1 byte
- list  -> b"L" + u64count + element frames in order; ONLY list — a tuple
  value is refused (no silent unification of distinct Python types)
- field -> b"F" + u64 namelen + name + value frame (nested ids embed their
  parent digest as a hex str field, which makes composition order explicit)
- ID    = sha256(field frame).hexdigest()

Refusal contract: the encoder never wraps silently.  An int outside the
signed 64-bit range raises struct.error; any unframed type raises
TypeError.  Every encoder refusal (TypeError/struct.error) is an
ADMISSION refusal for callers, never truncated or wrapped data.
"""
import hashlib
import struct


def _u64(n):
    return struct.pack("<Q", n)


def frame(v):
    if isinstance(v, bool):
        return b"B" + bytes([1 if v else 0])
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"s" + _u64(len(b)) + b
    if isinstance(v, int):
        return b"i" + struct.pack("<q", v)
    if isinstance(v, float):
        return b"f" + struct.pack("<d", v)
    if isinstance(v, bytes):
        return b"b" + _u64(len(v)) + v
    if isinstance(v, list):
        return (b"L" + _u64(len(v))
                + b"".join(frame(e) for e in v))
    raise TypeError("unframed type: %r" % (type(v),))


def frame_fields(pairs):
    out = []
    for name, value in pairs:
        nb = name.encode("utf-8")
        out.append(b"F" + _u64(len(nb)) + nb + frame(value))
    return b"".join(out)


def identity(pairs):
    """Canonical ID of an ordered field list."""
    return hashlib.sha256(frame_fields(pairs)).hexdigest()


def bits(v):
    """The evaluated IEEE-754 bits of a float (or float array)."""
    import numpy as np
    if isinstance(v, (int, float)):
        return struct.pack("<d", float(v))
    a = np.ascontiguousarray(v, dtype=np.float64)
    return a.tobytes()
