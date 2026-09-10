"""What the three iPIC3D reader variants share."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.readers._base import ReaderBase
from pypic.readers.ipic3d._config import IPic3DConfig, to_simulation_config
from pypic.readers.ipic3d._conserved import detect_conserved, load_ipic3d_auxiliary

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.containers import SimulationConfig, TabularData


class IPic3DReaderBase(ReaderBase):
    """Config translation and auxiliary data, common to every iPIC3D layout.

    Parameters
    ----------
    config : IPic3DConfig
        Parsed iPIC3D configuration.
    sim_config : SimulationConfig | None
        Merged run configuration; translated from *config* when omitted.
    """

    def __init__(
        self, config: IPic3DConfig, sim_config: SimulationConfig | None = None
    ) -> None:
        super().__init__(sim_config or to_simulation_config(config))
        self._config = config

    def available_auxiliary(self, path: Path) -> list[str]:
        """Names of the conserved-quantity and species diagnostics at *path*."""
        return detect_conserved(path)

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load one of the diagnostics listed by `available_auxiliary`."""
        return load_ipic3d_auxiliary(path, name)
