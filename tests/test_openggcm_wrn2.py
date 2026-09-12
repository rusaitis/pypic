"""Tests for the WRN2 lossy compression decoder."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from pypic.readers.openggcm._field_map import convert_fields_to_si
from pypic.readers.openggcm._wrn2 import (
    _build_exp_lut,
    _classify_rle_bytes,
    decode_rle,
    decompress_field,
    decompress_field_vectorized,
)

# Full-size OpenGGCM output (~10^8 values per field), far too large to
# commit. Maintainer-only: the vectorized-vs-original cross-validation that
# matters runs on synthetic data in TestDecompressFieldVectorized, and the
# committed subsample under tests/data/openggcm-small/ drives every other
# test in this file. Point PYPIC_OPENGGCM_RUN at a directory holding a real
# .3df run to enable it — the same variable scripts/generate_openggcm_fixture.py
# reads, so the fixture and the test that validates it share one source.
_OPENGGCM_RUN = os.environ.get("PYPIC_OPENGGCM_RUN")
_EXAMPLE_3DF = (
    Path(_OPENGGCM_RUN) / "gc012.3df.006300"
    if _OPENGGCM_RUN
    else Path(__file__).resolve().parent.parent / "examples" / "openggcm-run"
)


class TestDecodeRle:
    """Tests for the RLE line decoder."""

    def test_literal_values(self) -> None:
        """Each byte below 128 is a literal value.

        Fortran positions: i starts at -2, each literal does i+1 then
        stores.  First literal → i2(-1) = checksum.  Then i2(0), i2(1).
        """
        # i2(-1)=checksum, i2(0)=70, i2(1)=80
        # checksum = 33 + (70+80) % 92 = 33 + 58 = 91
        line = bytes([91, 70, 80, 32])
        data, n = decode_rle(line)
        assert n == 1
        assert data == [70, 80]

    def test_rle_run(self) -> None:
        """Byte > 127 signals a run of the next byte."""
        # Literal checksum, then RLE for data.
        # i2(-1)=checksum (literal), then 3 copies of 50 at i2(0..2).
        # checksum = 33 + (50*3) % 92 = 33 + 58 = 91
        # RLE byte: 170 + 3 = 173
        line = bytes([91, 173, 50, 32])
        data, n = decode_rle(line)
        assert n == 2
        assert data == [50, 50, 50]

    def test_checksum_failure_returns_error(self) -> None:
        """Invalid checksum produces n=-5."""
        # Deliberately wrong checksum (99 instead of 91)
        line = bytes([99, 70, 80, 32])
        data, n = decode_rle(line)
        assert n == -5
        assert data == []

    def test_real_compressed_line(self) -> None:
        """Decode a real line from the gc012.3df.006300 file.

        Byte sequence: [57, 234, 78, <space or end>]
        - 57 → literal, i = -1, i2(-1) = 57 (checksum)
        - 234 → RLE count = 234 - 170 = 64
        - 78 → repeated value: i2(0..63) = 78
        - space → end of record

        Checksum: sum(data) = 64*78 = 4992, 33 + 4992%92 = 57 ✓
        """
        line = bytes([57, 234, 78, 32])
        data, n = decode_rle(line)
        assert n == 63  # 0..63
        assert len(data) == 64
        assert all(v == 78 for v in data)

    def test_newline_in_line_is_skipped(self) -> None:
        """Newline bytes (10) are silently ignored."""
        # 91=checksum, 70=i2(0), 10=newline(skip), 80=i2(1), 32=end
        line = bytes([91, 70, 10, 80, 32])
        data, n = decode_rle(line)
        assert n == 1
        assert data == [70, 80]

    def test_empty_line_returns_error(self) -> None:
        """A line with only a space has no data."""
        line = bytes([32])
        _data, n = decode_rle(line)
        assert n == -5


class TestDecompressField:
    """Tests for the full WRN2 field decompressor."""

    def test_uniform_field(self) -> None:
        """When zmin == zmax, all values equal zmin."""
        result = decompress_field(iter([]), 100, 5.0, 5.0)
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, 5.0)

    def test_single_chunk(self) -> None:
        """Decode a single 64-value chunk with known values."""
        # Construct two RLE lines: checksum + 64 identical data values.
        # Line 1 (i1): checksum=57, data=[78]*64
        #   sum([78]*64) = 4992, 33 + 4992%92 = 57 ✓
        line1 = bytes([57, 234, 78, 32])
        # Line 2 (i2): checksum=117, data=[89]*64
        #   sum([89]*64) = 5696, 33 + 5696%92 = 117 ✓
        line2 = bytes([117, 234, 89, 32])

        zmin, zmax = -29.68665, 7.313355
        dzi = (zmax - zmin) / 4410.0
        # i1_adj = 78-33 = 45 < 47 → sign=+1, i1_adj stays 45
        # i2_adj = 89-33 = 56
        # i3 = 56 + 94*45 = 4286
        expected_val = math.exp(dzi * 4286 + zmin)

        result = decompress_field(iter([line1, line2]), 64, zmin, zmax)
        assert result.shape == (64,)
        np.testing.assert_allclose(result, expected_val, rtol=1e-10)

    def test_negative_values(self) -> None:
        """Values with i1 >= 47+33 = 80 produce negative output."""
        # i1 = 80 → i1_adj = 80-33 = 47 → negative, i1_adj -= 47 → 0
        # i2 = 50 → i2_adj = 50-33 = 17
        # i3 = 17 + 94*0 = 17
        # 1 value: nk=0
        # Line 1: checksum=33+80%92=113, data=[80]
        line1 = bytes([113, 80, 32])
        # Line 2: checksum=33+50%92=83, data=[50]
        line2 = bytes([83, 50, 32])

        zmin, zmax = -5.0, 5.0
        dzi = (zmax - zmin) / 4410.0
        expected_val = -math.exp(dzi * 17 + zmin)

        result = decompress_field(iter([line1, line2]), 1, zmin, zmax)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], expected_val, rtol=1e-10)

    def test_multiple_chunks(self) -> None:
        """65 values require two chunks (64 + 1)."""
        # All i1=78, all i2=89 for simplicity
        # Chunk 1: 64 values (nk=63)
        ck78_64 = 33 + ((78 * 64) % 92)  # 57
        ck89_64 = 33 + ((89 * 64) % 92)  # 117
        line1a = bytes([ck78_64, 234, 78, 32])
        line1b = bytes([ck89_64, 234, 89, 32])

        # Chunk 2: 1 value (nk=0)
        ck78_1 = 33 + (78 % 92)  # 111
        ck89_1 = 33 + (89 % 92)  # 122
        line2a = bytes([ck78_1, 78, 32])
        line2b = bytes([ck89_1, 89, 32])

        lines = [line1a, line1b, line2a, line2b]
        result = decompress_field(iter(lines), 65, -5.0, 5.0)
        assert result.shape == (65,)
        # All values should be equal since all i1/i2 are the same
        np.testing.assert_allclose(result, result[0], rtol=1e-10)


class TestBuildExpLut:
    """Tests for the exp lookup table builder."""

    def test_exp_lut_properties(self) -> None:
        lut = _build_exp_lut(-5.0, 5.0)
        assert lut.shape == (4418,)
        assert lut.dtype == np.float64
        # Monotonically increasing
        assert np.all(np.diff(lut) > 0)
        # Values match direct computation
        zmin, zmax = -29.68665, 7.313355
        lut2 = _build_exp_lut(zmin, zmax)
        dzi = (zmax - zmin) / 4410.0
        for i3 in [0, 100, 2205, 4417]:
            expected = math.exp(dzi * i3 + zmin)
            np.testing.assert_allclose(lut2[i3], expected, rtol=1e-14)


class TestClassifyRleBytes:
    """Tests for the byte classification function."""

    def test_all_literals(self) -> None:
        """Data bytes 33-126 are classified as literals."""
        raw = np.array([70, 80, 90, 10], dtype=np.uint8)
        is_literal, is_consumed = _classify_rle_bytes(raw)
        assert is_literal.tolist() == [True, True, True, False]
        assert is_consumed.tolist() == [False, False, False, False]

    def test_marker_and_consumed(self) -> None:
        """Byte > 127 is a marker; next byte is consumed."""
        raw = np.array([234, 78, 10], dtype=np.uint8)
        is_literal, is_consumed = _classify_rle_bytes(raw)
        assert is_literal.tolist() == [False, False, False]
        assert is_consumed.tolist() == [False, True, False]

    def test_space_not_literal(self) -> None:
        """Space (32) is structural, not a literal."""
        raw = np.array([70, 32, 80, 10], dtype=np.uint8)
        is_literal, _is_consumed = _classify_rle_bytes(raw)
        assert is_literal[0]
        assert not is_literal[1]  # space
        assert is_literal[2]

    def test_mixed_markers_and_literals(self) -> None:
        """Interleaved markers and literals are classified correctly."""
        # checksum=57(lit), marker=234 consumed=78, space=32, newline=10
        raw = np.array([57, 234, 78, 32, 10], dtype=np.uint8)
        is_literal, is_consumed = _classify_rle_bytes(raw)
        assert is_literal[0]  # 57 = literal
        assert not is_literal[1]  # 234 = marker
        assert not is_literal[2]  # 78 = consumed by marker
        assert not is_literal[3]  # 32 = space
        assert not is_literal[4]  # 10 = newline
        assert is_consumed[2]


def _make_wrn2_data(
    i1_vals: list[int],
    i2_vals: list[int],
) -> bytes:
    """Build WRN2-encoded bytes from known i1/i2 per-chunk values.

    Each chunk produces two lines terminated by space + newline.
    Supports a single chunk only (≤64 values).
    """

    def make_line(vals: list[int]) -> bytes:
        checksum = 33 + (sum(vals) % 92)
        # Build as: checksum literal, then all data literals, then space + newline
        parts = [checksum, *vals, 32, 10]
        return bytes(parts)

    return make_line(i1_vals) + make_line(i2_vals)


def _make_wrn2_rle_data(val1: int, val2: int, count: int = 64) -> bytes:
    """Build WRN2-encoded bytes using RLE for uniform chunks."""

    def make_rle_line(val: int, n: int) -> bytes:
        checksum = 33 + ((val * n) % 92)
        rle_byte = 170 + n
        return bytes([checksum, rle_byte, val, 32, 10])

    return make_rle_line(val1, count) + make_rle_line(val2, count)


class TestDecompressFieldVectorized:
    """Tests for the vectorized WRN2 field decompressor."""

    def test_single_chunk_matches_original(self) -> None:
        """Single 64-value chunk matches the line-by-line decoder."""
        zmin, zmax = -29.68665, 7.313355
        data = _make_wrn2_rle_data(78, 89, 64)

        # Original decoder
        line1 = bytes([57, 234, 78, 32])
        line2 = bytes([117, 234, 89, 32])
        expected = decompress_field(iter([line1, line2]), 64, zmin, zmax)

        # Vectorized decoder
        result = decompress_field_vectorized(data, 0, len(data), 64, zmin, zmax)
        np.testing.assert_array_equal(result, expected)

    def test_negative_values(self) -> None:
        """Values with i1 >= 80 (i1_adj >= 47) produce negative output."""
        zmin, zmax = -5.0, 5.0
        # i1=80 → negative, i2=50
        data = _make_wrn2_data([80], [50])

        # Original decoder
        line1 = bytes([113, 80, 32])
        line2 = bytes([83, 50, 32])
        expected = decompress_field(iter([line1, line2]), 1, zmin, zmax)

        result = decompress_field_vectorized(data, 0, len(data), 1, zmin, zmax)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_chunks(self) -> None:
        """65 values (two chunks: 64 + 1) match original decoder."""
        zmin, zmax = -5.0, 5.0

        # Chunk 1: 64 uniform values
        chunk1 = _make_wrn2_rle_data(78, 89, 64)
        # Chunk 2: 1 value (literal lines)
        chunk2 = _make_wrn2_data([78], [89])
        data = chunk1 + chunk2

        # Original decoder
        ck78_64 = 33 + ((78 * 64) % 92)
        ck89_64 = 33 + ((89 * 64) % 92)
        ck78_1 = 33 + (78 % 92)
        ck89_1 = 33 + (89 % 92)
        lines = [
            bytes([ck78_64, 234, 78, 32]),
            bytes([ck89_64, 234, 89, 32]),
            bytes([ck78_1, 78, 32]),
            bytes([ck89_1, 89, 32]),
        ]
        expected = decompress_field(iter(lines), 65, zmin, zmax)

        result = decompress_field_vectorized(data, 0, len(data), 65, zmin, zmax)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_literal_and_rle(self) -> None:
        """Chunk with mix of literal and RLE-encoded values."""
        zmin, zmax = -10.0, 10.0
        # 4 values: i1=[40, 50, 60, 70], i2=[35, 45, 55, 65]
        i1_vals = [40, 50, 60, 70]
        i2_vals = [35, 45, 55, 65]
        data = _make_wrn2_data(i1_vals, i2_vals)

        # Original decoder
        ck1 = 33 + (sum(i1_vals) % 92)
        ck2 = 33 + (sum(i2_vals) % 92)
        line1 = bytes([ck1, *i1_vals, 32])
        line2 = bytes([ck2, *i2_vals, 32])
        expected = decompress_field(iter([line1, line2]), 4, zmin, zmax)

        result = decompress_field_vectorized(data, 0, len(data), 4, zmin, zmax)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_offset_into_buffer(self) -> None:
        """Data at a non-zero offset in the buffer is decoded correctly."""
        zmin, zmax = -5.0, 5.0
        prefix = b"HEADER STUFF\n" * 5
        wrn2_data = _make_wrn2_rle_data(78, 89, 64)
        full_buf = prefix + wrn2_data + b"TRAILER\n"

        result = decompress_field_vectorized(
            full_buf, len(prefix), len(wrn2_data), 64, zmin, zmax
        )
        expected = decompress_field_vectorized(
            wrn2_data, 0, len(wrn2_data), 64, zmin, zmax
        )
        np.testing.assert_array_equal(result, expected)


class TestConvertFieldsPassthrough:
    """Unknown fields pass through convert_fields_to_si with native names."""

    def test_unknown_fields_pass_through(self) -> None:
        raw = {
            "vx": np.ones(10),
            "custom_field": np.full(10, 42.0),
        }
        si = convert_fields_to_si(raw)
        assert "V_1" in si
        np.testing.assert_array_equal(si["custom_field"], 42.0)
        # Unknown-only field is not converted
        raw2 = {"unknown_thing": np.array([1.0, 2.0, 3.0])}
        si2 = convert_fields_to_si(raw2)
        np.testing.assert_array_equal(si2["unknown_thing"], [1.0, 2.0, 3.0])


@pytest.mark.skipif(
    not _EXAMPLE_3DF.exists(),
    reason="set PYPIC_OPENGGCM_RUN to a full-size .3df run (not shipped)",
)
class TestRealData:
    """Integration tests against real .3df data."""

    def test_first_wrn2_block(self) -> None:
        """Read and verify the first WRN2 block from a real file."""
        with _EXAMPLE_3DF.open("rb") as f:
            # Skip 5 header lines
            for _ in range(5):
                f.readline()
            # Read first field's WRN2 data (vx, 98963200 values)
            result = decompress_field(iter(f), 98963200, -29.68665, 7.313355)
        assert result.shape == (98963200,)
        # vx spans a wide range: solar wind ~300-800 km/s but magnetotail
        # flows can be negative (earthward return flows)
        assert abs(result.min()) < 5000
        assert result.max() < 2000  # reasonable upper bound in km/s

    def test_vectorized_matches_original_on_real_data(self) -> None:
        """Cross-validate: vectorized decoder matches original on real data."""
        data = _EXAMPLE_3DF.read_bytes()

        # Parse header to find the first field's WRN2 data
        marker_pos = data.find(b"FIELD-3D-1")
        assert marker_pos != -1

        # Skip 4 header lines after marker
        pos = data.find(b"\n", marker_pos) + 1  # past marker line
        for _ in range(3):
            pos = data.find(b"\n", pos) + 1  # name, desc, dims

        # WRN2 header
        wrn2_end = data.find(b"\n", pos)
        wrn2_line = data[pos:wrn2_end]
        count = int(wrn2_line[4:12])
        zmin = float(wrn2_line[12:26])
        zmax = float(wrn2_line[26:40])
        data_start = wrn2_end + 1

        # Find end of this field's data (next marker or EOF)
        next_marker = data.find(b"FIELD-3D-1", data_start)
        data_end = next_marker if next_marker != -1 else len(data)

        # Vectorized
        result_vec = decompress_field_vectorized(
            data, data_start, data_end - data_start, count, zmin, zmax
        )

        # Original (line-by-line)
        with _EXAMPLE_3DF.open("rb") as f:
            for _ in range(5):
                f.readline()
            result_orig = decompress_field(iter(f), count, zmin, zmax)

        np.testing.assert_array_equal(result_vec, result_orig)
