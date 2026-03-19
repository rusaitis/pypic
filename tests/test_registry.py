"""Tests for the reader registry and auto-detection system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from pypic.readers._registry import (
    _REGISTRY,
    Simulation,
    open_simulation,
    register_reader,
    registered_readers,
    unregister_reader,
)
from pypic.readers.base import SimulationConfig, SimulationReader, TabularData

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
    path: Path,
    **_kwargs: Any,
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
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        register_reader("dup", lambda _: 0.1, _mock_factory)
        with caplog.at_level("WARNING"):
            register_reader("dup", lambda _: 0.9, _mock_factory)
        assert "Overwriting" in caplog.text


class TestOpenSimulationExplicitName:
    def test_open_with_reader_name(
        self,
        tmp_path: Path,
    ) -> None:
        called_with: list[Path] = []

        def factory(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            called_with.append(path)
            return _mock_factory(path)

        register_reader("named", lambda _: 0.5, factory)
        open_simulation(tmp_path, reader="named")
        assert called_with == [tmp_path]

    def test_open_with_unknown_name_raises(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            open_simulation(tmp_path, reader="nonexistent")

    def test_kwargs_forwarded_to_factory(
        self,
        tmp_path: Path,
    ) -> None:
        received_kwargs: dict[str, Any] = {}

        def factory(
            path: Path,
            **kwargs: Any,
        ) -> _Result:
            received_kwargs.update(kwargs)
            return _mock_factory(path)

        register_reader("kwarg_test", lambda _: 0.5, factory)
        open_simulation(
            tmp_path,
            reader="kwarg_test",
            normalization="custom",
        )
        assert received_kwargs["normalization"] == "custom"


class TestOpenSimulationCallable:
    def test_open_with_callable(self, tmp_path: Path) -> None:
        called: list[Path] = []

        def my_reader(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            called.append(path)
            return _mock_factory(path)

        open_simulation(tmp_path, reader=my_reader)
        assert called == [tmp_path]


class TestOpenSimulationAutoDetect:
    def test_highest_confidence_wins(
        self,
        tmp_path: Path,
    ) -> None:
        calls: list[str] = []

        def factory_a(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            calls.append("a")
            return _mock_factory(path)

        def factory_b(
            path: Path,
            **_kw: Any,
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
            FileNotFoundError,
            match="No registered reader",
        ):
            open_simulation(tmp_path)

    def test_alphabetical_tiebreak(
        self,
        tmp_path: Path,
    ) -> None:
        calls: list[str] = []

        def factory_a(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            calls.append("alpha")
            return _mock_factory(path)

        def factory_b(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            calls.append("beta")
            return _mock_factory(path)

        register_reader("beta", lambda _: 0.5, factory_b)
        register_reader("alpha", lambda _: 0.5, factory_a)

        open_simulation(tmp_path)
        assert calls == ["alpha"]

    def test_probe_exception_skipped(
        self,
        tmp_path: Path,
    ) -> None:
        """Probe that raises is skipped, not crash."""

        def broken_probe(_p: Path) -> float:
            raise RuntimeError("broken")

        calls: list[str] = []

        def good_factory(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            calls.append("good")
            return _mock_factory(path)

        register_reader("broken", broken_probe, _mock_factory)
        register_reader("good", lambda _: 0.5, good_factory)

        open_simulation(tmp_path)
        assert calls == ["good"]

    def test_string_path_accepted(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("str_test", lambda _: 0.5, _mock_factory)
        reader, _cfg = open_simulation(
            str(tmp_path),
            reader="str_test",
        )
        assert reader is not None


class TestCanReadIPic3D:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import can_read_confidence

        assert can_read_confidence(tmp_path) == 0.0

    def test_positive_detection(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import can_read_confidence

        (tmp_path / "GEM.inp").touch()
        assert can_read_confidence(tmp_path) >= 0.5

        # settings.hdf also triggers detection
        hdf_dir = tmp_path / "hdf_only"
        hdf_dir.mkdir()
        (hdf_dir / "settings.hdf").touch()
        assert can_read_confidence(hdf_dir) >= 0.4

        # H5hut files also trigger detection
        h5_dir = tmp_path / "h5hut_only"
        h5_dir.mkdir()
        (h5_dir / "GEM-Fields_000100.h5").touch()
        assert can_read_confidence(h5_dir) >= 0.3

    def test_not_a_directory(self, tmp_path: Path) -> None:
        from pypic.readers.ipic3d._probe import can_read_confidence

        f = tmp_path / "file.txt"
        f.touch()
        assert can_read_confidence(f) == 0.0


class TestCanReadBATSRUS:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import can_read_confidence

        assert can_read_confidence(tmp_path) == 0.0

    def test_positive_detection(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import can_read_confidence

        (tmp_path / "3d__n00000001.batl").touch()
        assert can_read_confidence(tmp_path) >= 0.5

        # PARAM.in also triggers detection
        param_dir = tmp_path / "param_only"
        param_dir.mkdir()
        (param_dir / "PARAM.in").touch()
        assert can_read_confidence(param_dir) >= 0.3

        # Header + IDL also triggers detection
        idl_dir = tmp_path / "idl_only"
        idl_dir.mkdir()
        (idl_dir / "3d__n00000001.h").touch()
        (idl_dir / "3d__n00000001_pe0000.idl").touch()
        assert can_read_confidence(idl_dir) >= 0.5

        # .out file also triggers detection
        out_dir = tmp_path / "out_only"
        out_dir.mkdir()
        (out_dir / "3d__n00000001.out").touch()
        assert can_read_confidence(out_dir) >= 0.3


class TestCanReadOpenGGCM:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import can_read_confidence

        assert can_read_confidence(tmp_path) == 0.0

    def test_grid_file_only(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import can_read_confidence

        (tmp_path / "grid.001.dat").touch()
        assert can_read_confidence(tmp_path) == 0.5

    def test_3df_file_only(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import can_read_confidence

        (tmp_path / "gc012.3df.006300").touch()
        assert can_read_confidence(tmp_path) == 0.5

    def test_full_match(self, tmp_path: Path) -> None:
        from pypic.readers.openggcm._probe import can_read_confidence

        (tmp_path / "grid.001.dat").touch()
        (tmp_path / "gc012.3df.006300").touch()
        assert can_read_confidence(tmp_path) == 1.0


class TestBuiltinReadersRegistered:
    def test_all_builtins_present(self) -> None:
        readers = registered_readers()
        assert "ipic3d" in readers
        assert "batsrus" in readers
        assert "openggcm" in readers


class TestSimulationFacade:
    def test_tuple_unpacking_and_properties(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("unpack", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="unpack")
        assert isinstance(sim, Simulation)
        reader, config = sim
        assert isinstance(reader, SimulationReader)
        assert isinstance(config, SimulationConfig)
        assert sim.model_name == sim.config.model_name
        assert sim.path == tmp_path

    def test_read_and_steps(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("read", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="read")
        sim.read(step=0)
        sim.reader.read_timestep.assert_called_once_with(
            tmp_path,
            0,
        )
        sim.reader.available_timesteps.return_value = [0, 10]
        first = sim.steps
        second = sim.steps
        assert first == [0, 10]
        assert first is second
        sim.reader.available_timesteps.assert_called_once()

    def test_auxiliary_delegation(
        self,
        tmp_path: Path,
    ) -> None:
        # Error path: basic reader without auxiliary support
        register_reader("noaux", lambda _: 0.5, _mock_factory)
        sim_basic = open_simulation(tmp_path, reader="noaux")
        with pytest.raises(TypeError, match="auxiliary data"):
            sim_basic.auxiliary("anything")

        # Happy path: reader with auxiliary support
        tab = TabularData(
            name="test_aux",
            columns={"x": np.array([1.0, 2.0])},
        )

        def factory_with_aux(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            reader = MagicMock(spec=SimulationReader)
            reader.available_auxiliary = MagicMock(
                return_value=["test_aux"],
            )
            reader.load_auxiliary = MagicMock(return_value=tab)
            reader.__class__ = type(
                "AuxReader",
                (),
                {
                    "available_auxiliary": lambda s, p: ["test_aux"],
                    "load_auxiliary": lambda s, p, n: tab,
                    "read_timestep": lambda s, p, st: None,
                    "available_timesteps": lambda s, p: [],
                },
            )
            config = MagicMock(spec=SimulationConfig)
            return reader, config

        register_reader("withaux", lambda _: 0.5, factory_with_aux)
        sim = open_simulation(tmp_path, reader="withaux")
        assert sim.auxiliary_names == ["test_aux"]
        result = sim.auxiliary("test_aux")
        assert result.name == "test_aux"
