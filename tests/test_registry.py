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
from pypic.readers.base import (
    FieldDataset,
    GridInfo,
    SimulationConfig,
    SimulationReader,
    TabularData,
    supports_selective_read,
)
from pypic.units import Normalization

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

    def test_selective_reader_dispatch(
        self,
        tmp_path: Path,
    ) -> None:
        """Simulation.read(fields=...) uses SelectiveReader protocol."""
        from pypic.coordinates.geometry import CARTESIAN

        grid = GridInfo(
            dimensions=(2,),
            spacing=(1.0,),
            origin=(0.0,),
            geometry=CARTESIAN,
        )
        norm = Normalization.identity()

        class FullReader:
            """Reader that supports selective I/O via fields= param."""

            def __init__(self) -> None:
                self.received_fields: set[str] | None = None

            def read_timestep(
                self,
                path: Path,
                step: int,
                *,
                fields: set[str] | None = None,
            ) -> FieldDataset:
                self.received_fields = fields
                data = {"B1": np.ones(2), "B2": np.ones(2), "rho_c": np.zeros(2)}
                if fields is not None:
                    data = {k: v for k, v in data.items() if k in fields}
                return FieldDataset.from_arrays(data, grid, norm)

            def available_timesteps(self, path: Path) -> list[int]:
                return [0]

        full_reader = FullReader()
        assert supports_selective_read(full_reader)

        cfg = SimulationConfig(
            model_name="test",
            model_type="PIC",
            grid=grid,
            normalization=norm,
        )
        sim = Simulation(full_reader, cfg, tmp_path)

        # Selective read should pass fields through to reader
        ds = sim.read(0, fields=["Bx"])
        assert full_reader.received_fields == {"B1"}
        assert ds.has_field("B1")

    def test_basic_reader_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        """Simulation.read(fields=...) falls back to select_fields for basic readers."""
        from pypic.coordinates.geometry import CARTESIAN

        grid = GridInfo(
            dimensions=(2,),
            spacing=(1.0,),
            origin=(0.0,),
            geometry=CARTESIAN,
        )
        norm = Normalization.identity()

        class BasicReader:
            """Reader without fields= support."""

            def read_timestep(self, path: Path, step: int) -> FieldDataset:
                return FieldDataset.from_arrays(
                    {"B1": np.ones(2), "B2": np.ones(2), "rho_c": np.zeros(2)},
                    grid,
                    norm,
                )

            def available_timesteps(self, path: Path) -> list[int]:
                return [0]

        basic_reader = BasicReader()
        assert isinstance(basic_reader, SimulationReader)
        assert not supports_selective_read(basic_reader)

        cfg = SimulationConfig(
            model_name="test",
            model_type="PIC",
            grid=grid,
            normalization=norm,
        )
        sim = Simulation(basic_reader, cfg, tmp_path)

        # Should read all then filter
        ds = sim.read(0, fields=["B1"])
        assert sorted(ds.field_names()) == ["B1"]


class TestBatsrusProbeFilter:
    def test_c_headers_not_detected(self, tmp_path: Path) -> None:
        """C/C++ headers (reader.h, config.h) must not trigger BATSRUS."""
        from pypic.readers.batsrus._probe import can_read_confidence

        (tmp_path / "reader.h").touch()
        (tmp_path / "config.h").touch()
        assert can_read_confidence(tmp_path) == 0.0

    def test_batsrus_headers_still_detected(self, tmp_path: Path) -> None:
        from pypic.readers.batsrus._probe import can_read_confidence

        (tmp_path / "3d__n00000001.h").touch()
        assert can_read_confidence(tmp_path) >= 0.3


class TestProbeResultDiagnostics:
    def test_failure_message_contains_all_readers(
        self,
        tmp_path: Path,
    ) -> None:
        _REGISTRY.clear()
        register_reader("alpha", lambda _: 0.0, _mock_factory)
        register_reader("beta", lambda _: 0.0, _mock_factory)

        with pytest.raises(FileNotFoundError, match="alpha") as exc_info:
            open_simulation(tmp_path)

        msg = str(exc_info.value)
        assert "alpha" in msg
        assert "beta" in msg
        assert "0.00" in msg

    def test_probe_exception_shown_in_message(
        self,
        tmp_path: Path,
    ) -> None:
        _REGISTRY.clear()

        def broken_probe(_p: Path) -> float:
            raise RuntimeError("disk on fire")

        register_reader("broken", broken_probe, _mock_factory)

        with pytest.raises(FileNotFoundError) as exc_info:
            open_simulation(tmp_path)
        assert "RuntimeError" in str(exc_info.value)


class TestFactoryFallback:
    def test_failing_factory_falls_through(
        self,
        tmp_path: Path,
    ) -> None:
        """If highest-confidence reader's factory crashes, try next."""
        calls: list[str] = []

        def failing_factory(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            raise RuntimeError("corrupt config")

        def working_factory(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            calls.append("working")
            return _mock_factory(path)

        register_reader("high", lambda _: 0.9, failing_factory)
        register_reader("low", lambda _: 0.5, working_factory)

        sim = open_simulation(tmp_path)
        assert calls == ["working"]
        assert isinstance(sim, Simulation)

    def test_all_factories_fail_raises_exception_group(
        self,
        tmp_path: Path,
    ) -> None:
        def failing_a(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            raise ValueError("bad A")

        def failing_b(
            path: Path,
            **_kw: Any,
        ) -> _Result:
            raise RuntimeError("bad B")

        register_reader("fa", lambda _: 0.8, failing_a)
        register_reader("fb", lambda _: 0.5, failing_b)

        with pytest.raises(ExceptionGroup) as exc_info:
            open_simulation(tmp_path)
        assert len(exc_info.value.exceptions) == 2


class TestProbeResultsOnSimulation:
    def test_auto_detected_has_probe_results(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("probe_a", lambda _: 0.5, _mock_factory)
        register_reader("probe_b", lambda _: 0.0, _mock_factory)
        sim = open_simulation(tmp_path)
        assert sim.probe_results is not None
        names = {pr.name for pr in sim.probe_results}
        assert "probe_a" in names
        assert "probe_b" in names

    def test_explicit_reader_has_no_probe_results(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("explicit", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="explicit")
        assert sim.probe_results is None


class TestDescribe:
    def test_contains_key_info(
        self,
        tmp_path: Path,
    ) -> None:
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.units import SpeciesInfo

        grid = GridInfo(
            dimensions=(128, 64, 64),
            spacing=(0.5, 0.5, 0.5),
            origin=(0.0, 0.0, 0.0),
            geometry=CARTESIAN,
        )
        cfg = SimulationConfig(
            model_name="iPIC3D",
            model_type="PIC",
            grid=grid,
            normalization=Normalization.identity(),
            species=(
                SpeciesInfo(name="electrons", charge=-1.0, mass=1 / 256),
                SpeciesInfo(name="ions", charge=1.0, mass=1.0),
            ),
        )
        reader = MagicMock(spec=SimulationReader)
        sim = Simulation(reader, cfg, tmp_path)
        desc = sim.describe()
        assert "iPIC3D" in desc
        assert "PIC" in desc
        assert "128 x 64 x 64" in desc
        assert "electrons" in desc
        assert "ions" in desc

    def test_steps_shown_only_when_cached(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("desc_test", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="desc_test")
        assert "Steps" not in sim.describe()

        sim.reader.available_timesteps.return_value = [0, 100, 200]
        _ = sim.steps  # trigger caching
        assert "Steps" in sim.describe()
        assert "3 [0..200]" in sim.describe()


class TestRefreshSteps:
    def test_refresh_clears_cache(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("refresh", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="refresh")
        sim.reader.available_timesteps.return_value = [0, 10]
        assert sim.steps == [0, 10]

        sim.reader.available_timesteps.return_value = [0, 10, 20]
        # Cached — still old
        assert sim.steps == [0, 10]
        # Refresh — gets new
        assert sim.refresh_steps() == [0, 10, 20]


class TestFirstLastStep:
    def test_values(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("fl", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="fl")
        sim.reader.available_timesteps.return_value = [100, 200, 300]
        assert sim.first_step == 100
        assert sim.last_step == 300

    def test_triggers_lazy_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        register_reader("lazy", lambda _: 0.5, _mock_factory)
        sim = open_simulation(tmp_path, reader="lazy")
        sim.reader.available_timesteps.return_value = [0, 50]
        assert sim._steps is None  # not yet cached
        _ = sim.first_step
        assert sim._steps is not None
