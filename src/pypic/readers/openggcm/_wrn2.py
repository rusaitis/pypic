"""WRN2 lossy compression decoder for OpenGGCM .3df files.

The WRN2 scheme encodes floating-point arrays into two interleaved byte
streams per 64-value chunk.  Each byte stream is further compressed using
a simple run-length encoding (RLE) where bytes > 127 signal a repeated
value.

The encoding stores ``sign * exp(dzi * i3 + zmin)`` where ``i3`` is a
12.5-bit index reconstructed from two 7-bit halves, providing roughly
4 decimal digits of precision.  See Raeder (2003), OpenGGCM documentation.

Reference Fortran: ``box_IPIC05_gc012_0145_HR03_Leo.f`` subroutines
``wrndec`` (line 765) and ``rdn2`` (line 884).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from pypic.types import FloatArray


# WRN2 byte encoding, from the Fortran ``wrndec`` / ``rdn2`` pair. Every
# value below is a property of the on-disk format, not a tunable, and each
# is read by both the scalar and the vectorized path.
_NEWLINE_BYTE = 10
_EOR_BYTE = 32  # space terminates a record
_MARKER_MAX = 127  # bytes above this are RLE markers
_RLE_BASE = 170  # a marker byte encodes (byte - 170) repeats
_ASCII_BASE = 33  # printable-range offset carried by every data byte
_RADIX = 94  # distinct values per 7-bit half
_SIGN_FLAG = 47  # set in the high half for negative values
_CHECKSUM_MOD = 92
_CHUNK = 64  # values per chunk, two RLE lines each
_I3_COUNT = _RADIX * _SIGN_FLAG  # 4418 reconstructible 12.5-bit indices
_I3_SPAN = _I3_COUNT - 8  # max usable 12.5-bit index


# REFERENCE IMPL: the readable scalar decoder. Production reads go through
# decompress_field_vectorized; this one is the oracle it is cross-validated
# against in tests/test_openggcm_wrn2.py, so it stays.
def decode_rle(line: bytes) -> tuple[list[int], int]:
    r"""Decode one RLE-encoded WRN2 line into integer values.

    The Fortran subroutine ``wrndec`` reads a 72-character record and
    decodes it as follows:

    * Byte == 32 (space): end of record.
    * Byte > 127: run-length.  Count = byte - 170; the *next* byte is
      repeated *count* times.
    * Otherwise: literal single value.

    The first decoded value is internal (position -2 in Fortran), the
    second is the checksum (position -1), and the remaining 0..n are
    the data.

    Parameters
    ----------
    line : bytes
        Raw bytes of one line (newline stripped or present—it's ignored).

    Returns
    -------
    data : list[int]
        Decoded data values (positions 0..n in Fortran notation).
    n : int
        Number of data values minus one (Fortran convention: 0..n,
        so *n* + 1 values total).  Returns -5 on decode error.
    """
    # Track Fortran position i, starting at -2.  The first i++ makes
    # i = -1 (checksum slot).  Position i2(-2) is never written.
    i = -2
    i2: dict[int, int] = {}
    j = 0
    length = len(line)
    while j < length:
        ir = line[j]
        if ir == _NEWLINE_BYTE:
            j += 1
            continue
        if ir == _EOR_BYTE:
            break
        if ir > _MARKER_MAX:  # RLE: repeat the next byte
            count = ir - _RLE_BASE
            j += 1
            if j >= length:
                return [], -5
            val = line[j]
            for _ in range(count):
                i += 1
                i2[i] = val
        else:
            i += 1
            i2[i] = ir
        j += 1

    n = i  # Fortran: data positions are 0..n
    if n >= _CHUNK:
        return [], -5
    if n < 0:
        return [], -5

    data = [i2.get(k, 0) for k in range(n + 1)]

    # Validate checksum: i2(-1) == 33 + sum(data) % 92
    checksum_byte = i2.get(-1, 0)
    ick = sum(data)
    expected = _ASCII_BASE + (ick % _CHECKSUM_MOD)
    if checksum_byte != expected:
        return [], -5

    return data, n


# REFERENCE IMPL: scalar counterpart to decompress_field_vectorized, kept as
# the oracle for the cross-validation tests rather than for production reads.
def decompress_field(
    lines: Iterator[bytes], count: int, zmin: float, zmax: float
) -> FloatArray:
    r"""Decompress a WRN2-encoded field into a flat float64 array.

    Implements Fortran subroutine ``rdn2``.  Processes *count* values
    in chunks of 64, reading two RLE lines per chunk.

    Parameters
    ----------
    lines : Iterator[bytes]
        Line iterator positioned after the WRN2 header.
    count : int
        Total number of values to decode.
    zmin, zmax : float
        Log-space range from the WRN2 header.

    Returns
    -------
    FloatArray
        Decoded values, shape ``(count,)``.
    """
    if zmin == zmax:
        return np.full(count, zmin, dtype=np.float64)

    result = np.empty(count, dtype=np.float64)
    dzi = (zmax - zmin) / _I3_SPAN
    pos = 0

    for k in range(0, count, _CHUNK):
        nk = min(_CHUNK - 1, count - k - 1)  # Fortran: min0(63, n-k)
        expected_count = nk + 1

        line1 = next(lines)
        i1_data, n1 = decode_rle(line1)
        if n1 != nk:
            msg = f"WRN2 decode error: expected n={nk}, got {n1} (line 1, offset {k})"
            raise ValueError(msg)

        line2 = next(lines)
        i2_data, n2 = decode_rle(line2)
        if n2 != nk:
            msg = f"WRN2 decode error: expected n={nk}, got {n2} (line 2, offset {k})"
            raise ValueError(msg)

        # Vectorized value reconstruction
        a1 = np.array(i1_data[:expected_count], dtype=np.int32) - _ASCII_BASE
        a2 = np.array(i2_data[:expected_count], dtype=np.int32) - _ASCII_BASE

        sign = np.ones(expected_count, dtype=np.float64)
        negative = a1 >= _SIGN_FLAG
        sign[negative] = -1.0
        a1[negative] -= _SIGN_FLAG

        i3 = a2 + _RADIX * a1
        vals = sign * np.exp(dzi * i3 + zmin)

        result[pos : pos + expected_count] = vals
        pos += expected_count

    return result


def _build_exp_lut(zmin: float, zmax: float) -> FloatArray:
    r"""Pre-compute ``exp(dzi * i3 + zmin)`` for all valid i3 values.

    The WRN2 12.5-bit index ``i3`` ranges from 0 to 4417 (94*47-1).
    Building the LUT once and indexing into it avoids calling ``np.exp``
    on 99M values per field.

    Parameters
    ----------
    zmin, zmax : float
        Log-space range from the WRN2 header.

    Returns
    -------
    FloatArray
        Shape ``(4418,)`` lookup table.
    """
    dzi = (zmax - zmin) / _I3_SPAN
    i3 = np.arange(_I3_COUNT, dtype=np.float64)
    return np.exp(dzi * i3 + zmin)


def _classify_rle_bytes(
    raw: NDArray[np.uint8],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    r"""Classify each byte in a raw WRN2 data region.

    WRN2 data bytes are in range 33-126 (``33 + value`` where value is
    0-93).  Marker bytes (RLE counts) are always > 127.  Newlines (10)
    and spaces (32, end of record) are structural.

    Returns two boolean masks over *raw*:

    - **is_literal** — True for bytes that are decoded values (not markers,
      not consumed-by-markers, not newlines, not spaces).
    - **is_consumed** — True for the byte immediately after a marker
      (the value to be repeated).

    Together with the marker mask (``raw > 127``), these partition all
    non-structural bytes into emitters: markers expand to
    ``byte - 170`` copies of the consumed byte; literals emit themselves
    once.

    Parameters
    ----------
    raw : NDArray[np.uint8]
        Raw byte array from the WRN2 data region.

    Returns
    -------
    is_literal : NDArray[np.bool_]
        True at positions that are literal data values.
    is_consumed : NDArray[np.bool_]
        True at positions that are the repeated value after a marker.
    """
    is_marker = raw > _MARKER_MAX
    is_newline = raw == _NEWLINE_BYTE
    is_space = raw == _EOR_BYTE

    # The byte after a marker is consumed (repeated value), unless the
    # marker is the last byte before a newline boundary — but in practice
    # this doesn't happen in well-formed WRN2 data.
    is_consumed = np.zeros(len(raw), dtype=np.bool_)
    marker_indices = np.flatnonzero(is_marker)
    consumed_indices = marker_indices + 1
    # Filter: must be in bounds and not a newline (line boundary)
    valid = consumed_indices < len(raw)
    consumed_indices = consumed_indices[valid]
    is_consumed[consumed_indices] = True

    is_literal = ~is_marker & ~is_consumed & ~is_newline & ~is_space
    return is_literal, is_consumed


def decompress_field_vectorized(
    data: bytes | bytearray,
    offset: int,
    length: int,
    count: int,
    zmin: float,
    zmax: float,
) -> FloatArray:
    r"""Decompress a WRN2-encoded field using bulk NumPy operations.

    Replaces per-line ``decode_rle`` + ``decompress_field`` with a
    single-pass vectorized approach:

    1. Classify bytes (marker / consumed / literal / structural)
    2. Build repeat counts and emitted values
    3. ``np.repeat`` to expand the full decoded stream
    4. Deinterleave odd/even lines → i1, i2 arrays
    5. LUT gather for final ``sign * exp(dzi * i3 + zmin)``

    Parameters
    ----------
    data : bytes | bytearray
        The full file contents (or a memoryview-compatible buffer).
    offset : int
        Byte offset where WRN2 data lines begin (after the header).
    length : int
        Number of bytes in the WRN2 data region.
    count : int
        Total number of values to decode.
    zmin, zmax : float
        Log-space range from the WRN2 header.

    Returns
    -------
    FloatArray
        Decoded values, shape ``(count,)``.
    """
    if zmin == zmax:
        return np.full(count, zmin, dtype=np.float64)

    # Phase 1: extract raw bytes as uint8 array
    raw = np.frombuffer(data, dtype=np.uint8, count=length, offset=offset)

    # Phase 2: classify bytes
    is_marker = raw > _MARKER_MAX
    is_literal, _is_consumed = _classify_rle_bytes(raw)

    # Phase 3: build emit values and repeat counts for non-structural bytes
    # Emitters are: markers (which emit the consumed byte N times) and
    # literals (which emit themselves once).
    is_emitter = is_marker | is_literal
    emitter_idx = np.flatnonzero(is_emitter)

    # For markers: the emitted value is raw[idx+1], count is raw[idx]-170
    # For literals: the emitted value is raw[idx], count is 1
    emit_values = np.empty(len(emitter_idx), dtype=np.uint8)
    repeat_counts = np.ones(len(emitter_idx), dtype=np.int32)

    emitter_is_marker = is_marker[emitter_idx]
    marker_emitters = emitter_idx[emitter_is_marker]
    literal_emitters = emitter_idx[~emitter_is_marker]

    emit_values[emitter_is_marker] = raw[marker_emitters + 1]
    repeat_counts[emitter_is_marker] = raw[marker_emitters].astype(np.int32) - _RLE_BASE
    emit_values[~emitter_is_marker] = raw[literal_emitters]

    # Phase 4: expand to full decoded stream
    decoded = np.repeat(emit_values, repeat_counts)

    # Phase 5: find newline positions to split into lines
    newline_pos = np.flatnonzero(raw == _NEWLINE_BYTE)
    n_lines = len(newline_pos)

    # Each line's decoded values sit between newlines.  We need to map
    # each emitter to its line index, then split decoded values per line.
    # Assign each emitter byte to a line via searchsorted on newline positions.
    emitter_line = np.searchsorted(newline_pos, emitter_idx, side="right")

    # Build per-line boundaries in the decoded array
    # cumulative counts tell us where each emitter's expansion starts
    cum_counts = np.empty(len(repeat_counts) + 1, dtype=np.int64)
    cum_counts[0] = 0
    np.cumsum(repeat_counts, out=cum_counts[1:])

    # For each line, find the range in decoded[]
    # line_starts[i] = cum_counts[first emitter in line i]
    # line_ends[i] = cum_counts[last emitter in line i + 1]
    line_starts = np.empty(n_lines, dtype=np.int64)
    line_ends = np.empty(n_lines, dtype=np.int64)

    # Find first and last emitter index for each line
    if n_lines > 0:
        # For each line L, find the first emitter where emitter_line == L
        # and the last emitter where emitter_line == L.
        # Use searchsorted on sorted emitter_line (which is already sorted
        # since emitter_idx is sorted and newline_pos is sorted).
        line_first = np.searchsorted(emitter_line, np.arange(n_lines), side="left")
        line_last = np.searchsorted(emitter_line, np.arange(n_lines), side="right")

        line_starts[:] = cum_counts[line_first]
        line_ends[:] = cum_counts[line_last]

    # Phase 6: deinterleave lines into i1/i2 (vectorized)
    # Each line decodes to: checksum (pos -1), then data (pos 0..nk).
    # Lines come in pairs: (0, 1) → chunk 0, (2, 3) → chunk 1, ...
    n_chunks = (count + _CHUNK - 1) // _CHUNK
    last_chunk_size = count - (n_chunks - 1) * _CHUNK  # 1..64

    expected_lines = n_chunks * 2
    if n_lines < expected_lines:
        msg = f"WRN2: not enough data lines (expected {expected_lines}, got {n_lines})"
        raise ValueError(msg)

    i1_flat = np.empty(count, dtype=np.int32)
    i2_flat = np.empty(count, dtype=np.int32)

    # Full chunks (exactly 64 values each) — all except the last
    n_full = n_chunks - 1
    if n_full > 0:
        i1_starts = line_starts[0 : n_full * 2 : 2] + 1  # skip checksum
        i2_starts = line_starts[1 : n_full * 2 : 2] + 1
        offsets = np.arange(_CHUNK, dtype=np.int64)
        i1_src = (i1_starts[:, np.newaxis] + offsets).ravel()
        i2_src = (i2_starts[:, np.newaxis] + offsets).ravel()
        i1_flat[: n_full * _CHUNK] = decoded[i1_src]
        i2_flat[: n_full * _CHUNK] = decoded[i2_src]

    # Last chunk (1..64 values)
    s1 = line_starts[(n_chunks - 1) * 2] + 1
    s2 = line_starts[(n_chunks - 1) * 2 + 1] + 1
    i1_flat[n_full * _CHUNK :] = decoded[s1 : s1 + last_chunk_size]
    i2_flat[n_full * _CHUNK :] = decoded[s2 : s2 + last_chunk_size]

    # Phase 7: reconstruct values using LUT
    lut = _build_exp_lut(zmin, zmax)

    a1 = i1_flat - _ASCII_BASE
    a2 = i2_flat - _ASCII_BASE

    sign = np.ones(count, dtype=np.float64)
    negative = a1 >= _SIGN_FLAG
    sign[negative] = -1.0
    a1[negative] -= _SIGN_FLAG

    i3 = a2 + _RADIX * a1
    return sign * lut[i3]  # type: ignore[no-any-return]
