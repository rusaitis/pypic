"""Simulation data readers and auto-detection registry."""

from pypic.readers._base import ReaderBase
from pypic.readers._config_helpers import merge_simulation_toml
from pypic.readers._protocols import (
    AuxiliaryDataReader,
    ParticleDataReader,
    SimulationReader,
    score_signals,
    supports_selective_read,
)
from pypic.readers._registry import (
    ProbeResult,
    ReaderEntry,
    Simulation,
    open_simulation,
    register_reader,
    registered_readers,
    unregister_reader,
)
from pypic.readers._simple import SimpleReader, open_simple
from pypic.readers._simple import _open_reader as _open_simple_reader
from pypic.readers._simple import can_read_confidence as _simple_confidence
from pypic.readers.batsrus import (
    BATSRUSConfig,
    BATSRUSOutputFormat,
    BATSRUSReader,
    open_batsrus,
    parse_param_in,
)
from pypic.readers.batsrus import can_read_confidence as _batsrus_confidence
from pypic.readers.config import load_config
from pypic.readers.ipic3d import (
    ConservedQuantities,
    IPic3DConfig,
    IPic3DH5hutReader,
    IPic3DParallelReader,
    IPic3DSerialReader,
    conserved_to_tabular,
    detect_particle_steps,
    load_conserved_quantities,
    open_ipic3d,
    parse_inp,
    read_phdf5_particles,
)
from pypic.readers.ipic3d import can_read_confidence as _ipic3d_confidence
from pypic.readers.openggcm import (
    OpenGGCMGrid,
    OpenGGCMReader,
    open_openggcm,
    parse_grid_file,
)
from pypic.readers.openggcm import can_read_confidence as _openggcm_confidence

# The built-in readers register here, in one place, so the registry never
# depends on which subpackage happened to be imported first.
register_reader("batsrus", _batsrus_confidence, open_batsrus)
register_reader("ipic3d", _ipic3d_confidence, open_ipic3d)
register_reader("openggcm", _openggcm_confidence, open_openggcm)
register_reader("simple", _simple_confidence, _open_simple_reader)

__all__ = [
    "AuxiliaryDataReader",
    "BATSRUSConfig",
    "BATSRUSOutputFormat",
    "BATSRUSReader",
    "ConservedQuantities",
    "IPic3DConfig",
    "IPic3DH5hutReader",
    "IPic3DParallelReader",
    "IPic3DSerialReader",
    "OpenGGCMGrid",
    "OpenGGCMReader",
    "ParticleDataReader",
    "ProbeResult",
    "ReaderBase",
    "ReaderEntry",
    "SimpleReader",
    "Simulation",
    "SimulationReader",
    "conserved_to_tabular",
    "detect_particle_steps",
    "load_config",
    "load_conserved_quantities",
    "merge_simulation_toml",
    "open_batsrus",
    "open_ipic3d",
    "open_openggcm",
    "open_simple",
    "open_simulation",
    "parse_grid_file",
    "parse_inp",
    "parse_param_in",
    "read_phdf5_particles",
    "register_reader",
    "registered_readers",
    "score_signals",
    "supports_selective_read",
    "unregister_reader",
]
