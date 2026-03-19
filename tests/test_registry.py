"""Tests for the reader registry and auto-detection system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from pypic.readers._registry import (
    _REGISTRY,
    ReaderEntry,
    open_simulation,
    register_reader,
    registered_readers,
    unregister_reader,
)
from pypic.readers.base import SimulationConfig, SimulationReader

if TYPE_CHECKING:
    from pathlib import Path

# Type alias to keep factory signatures under 88 chars
type _Result = tuple[SimulationReader, SimulationConfig]


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # type: ignore[misc]
    """Save and restore the global registry around each test."""
    saved = dict(_REGISTRY)
    yield  # type: ignore[misc]
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _mock_factory(
    path: Path, **_kwargs: Any,
) -> _Result:
    """Dummy factory that returns mocks."""
    reader = MagicMock(spec=SimulationReader)
    config = MagicMock(spec=SimulationConfig)
    return reader, config


class TestRegisterUnregister:
    def test_register_and_lookup(self) -> None:
        register_reader("test_reader", lambda _: 0.5, _mock_factory)
        readers = registered_readers()
        assert "test_reader" in readers
        assert readers["test_reader"].name == "test_reader"

    def test_unregister_removes_entry(self) -> None:
        register_reader("temp", lambda _: 0.5, _mock_factory)
        assert "temp" in registered_readers()
        unregister_reader("temp")
        assert "temp" not in registered_readers()

    def test_unregister_nonexistent_raises(self) -> None:
        with pytest.raises(KeyError, match="no_such_reader"):
            unregister_reader("no_such_reader")

    def test_register_overwrites_with_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        register_reader("dup", lambda _: 0.1, _mock_factory)
        with caplog.at_level("WARNING"):
            register_reader("dup", lambda _: 0.9, _mock_factory)
        assert "Overwriting" in caplog.text

    def test_registered_readers_is_readonly(self) -> None:
        readers = registered_readers()
        with pytest.raises(TypeError):
            readers["hack"] = ReaderEntry(  # type: ignore[index]
                "hack", lambda _: 1.0, _mock_factory,
            )


class TestOpenSimulationExplicitName:
    def test_open_with_reader_name(
        self, tmp_path: Path,
    ) -> None:
        called_with: list[Path] = []

        def factory(
            path: Path, **_kw: Any,
        ) -> _Result:
            called_with.append(path)
            return _mock_factory(path)

        register_reader("named", lambda _: 0.5, factory)
        open_simulation(tmp_path, reader="named")
        assert called_with == [tmp_path]

    def test_open_with_unknown_name_raises(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            open_simulation(tmp_path, reader="nonexistent")

    def test_kwargs_forwarded_to_factory(
        self, tmp_path: Path,
    ) -> None:
        received_kwargs: dict[str, Any] = {}

        def factory(
            path: Path, **kwargs: Any,
        ) -> _Result:
            received_kwargs.update(kwargs)
            return _mock_factory(path)

        register_reader("kwarg_test", lambda _: 0.5, factory)
        open_simulation(
            tmp_path, reader="kwarg_test", normalization="custom",
        )
        assert received_kwargs["normalization"] == "custom"


class TestOpenSimulationCallable:
    def test_open_with_callable(self, tmp_path: Path) -> None:
        called: list[Path] = []

        def my_reader(
            path: Path, **_kw: Any,
        ) -> _Result:
            called.append(path)
            return _mock_factory(path)

        open_simulation(tmp_path, reader=my_reader)
        assert called == [tmp_path]

    def test_callable_receives_kwargs(
        self, tmp_path: Path,
    ) -> None:
        received: dict[str, Any] = {}

        def my_reader(
            path: Path, **kwargs: Any,
        ) -> _Result:
            received.update(kwargs)
            return _mock_factory(path)

        open_simulation(tmp_path, reader=my_reader, foo="bar")
        assert received["foo"] == "bar"


class TestOpenSimulationAutoDetect:
    def test_highest_confidence_wins(
        self, tmp_path: Path,
    ) -> None:
        calls: list[str] = []

        def factory_a(
            path: Path, **_kw: Any,
        ) -> _Result:
            calls.append("a")
            return _mock_factory(path)

        def factory_b(
            path: Path, **_kw: Any,
        ) -> _Result:
            calls.append("b")
            return _mock_factory(path)

        register_reader("low", lambda _: 0.3, factory_a)
        register_reader("high", lambda _: 0.9, factory_b)

        open_simulation(tmp_path)
        assert calls == ["b"]

    def test_no_match_raises(self, tmp_path: Path) -> None:
        _REGISTRY.clear()
        register_reader("zero", lambda _: 0.0, _mock_factory)

        with pytest.raises(
            FileNotFoundError, match="No registered reader",
        ):
            open_simulation(tmp_path)

    def test_alphabetical_tiebreak(
        self, tmp_path: Path,
    ) -> None:
        calls: list[str] = []

        def factory_a(
            path: Path, **_kw: Any,
        ) -> _Result:
            calls.append("alpha")
            return _mock_factory(path)

        def factory_b(
            path: Path, **_kw: Any,
        ) -> _Result:
            calls.append("beta")
            return _mock_factory(path)

        register_reader("beta", lambda _: 0.5, factory_b)
        register_reader("alpha", lambda _: 0.5, factory_a)

        open_simulation(tmp_path)
        assert calls == ["alpha"]

    def test_probe_exception_skipped(
        self, tmp_path: Path,
    ) -> None:
        """Probe that raises is skipped, not crash."""

        def broken_probe(_p: Path) -> float:
            raise RuntimeError("broken")

        calls: list[str] = []

        def good_factory(
            path: Path, **_kw: Any,
        ) -> _Result:
            calls.append("good")
            return _mock_factory(path)

        register_reader("broken", broken_probe, _mock_factory)
        register_reader("good", lambda _: 0.5, good_factory)

        open_simulation(tmp_path)
        assert calls == ["good"]

    def test_string_path_accepted(
        self, tmp_path: Path,
    ) -> None:
        register_reader("str_test", lambda _: 0.5, _mock_factory)
        reader, _cfg = open_simulation(
            str(tmp_path), reader="str_test",
        )
        assert reader is not None


class TestProbeIPic3D:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import probe

        assert probe(tmp_path) == 0.0

    def test_inp_file(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import probe

        (tmp_path / "GEM.inp").touch()
        assert probe(tmp_path) >= 0.5

    def test_settings_hdf(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import probe

        (tmp_path / "settings.hdf").touch()
        assert probe(tmp_path) >= 0.4

    def test_h5hut_files(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import probe

        (tmp_path / "GEM-Fields_000100.h5").touch()
        assert probe(tmp_path) >= 0.3

    def test_not_a_directory(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import probe

        f = tmp_path / "file.txt"
        f.touch()
        assert probe(f) == 0.0


class TestProbeBATSRUS:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import probe

        assert probe(tmp_path) == 0.0

    def test_batl_file(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import probe

        (tmp_path / "3d__n00000001.batl").touch()
        assert probe(tmp_path) >= 0.5

    def test_param_in(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import probe

        (tmp_path / "PARAM.in").touch()
        assert probe(tmp_path) >= 0.3

    def test_header_plus_idl(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import probe

        (tmp_path / "3d__n00000001.h").touch()
        (tmp_path / "3d__n00000001_pe0000.idl").touch()
        assert probe(tmp_path) >= 0.5

    def test_out_file(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import probe

        (tmp_path / "3d__n00000001.out").touch()
        assert probe(tmp_path) >= 0.3


class TestProbeOpenGGCM:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import probe

        assert probe(tmp_path) == 0.0

    def test_grid_file_only(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import probe

        (tmp_path / "grid.001.dat").touch()
        assert probe(tmp_path) == 0.5

    def test_3df_file_only(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import probe

        (tmp_path / "gc012.3df.006300").touch()
        assert probe(tmp_path) == 0.5

    def test_full_match(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import probe

        (tmp_path / "grid.001.dat").touch()
        (tmp_path / "gc012.3df.006300").touch()
        assert probe(tmp_path) == 1.0


class TestBuiltinReadersRegistered:
    def test_all_builtins_present(self) -> None:
        readers = registered_readers()
        assert "ipic3d" in readers
        assert "batsrus" in readers
        assert "openggcm" in readers

    def test_builtin_entries_have_probes(self) -> None:
        for name, entry in registered_readers().items():
            assert callable(entry.probe), f"{name} probe"
            assert callable(entry.factory), f"{name} factory"


class TestBATSRUSOutputFormat:
    def test_values(self) -> None:
        from pypic.readers.batsrus import BATSRUSOutputFormat

        assert BATSRUSOutputFormat.HDF5 == "hdf5"
        assert BATSRUSOutputFormat.IDL == "idl"
        assert BATSRUSOutputFormat.OUT == "out"

    def test_is_str(self) -> None:
        from pypic.readers.batsrus import BATSRUSOutputFormat

        assert isinstance(BATSRUSOutputFormat.HDF5, str)
