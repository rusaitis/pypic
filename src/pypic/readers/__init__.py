"""Simulation data readers and the FieldDataset container."""

from pypic.readers._registry import (
    ReaderEntry,
    open_simulation,
    register_reader,
    registered_readers,
    unregister_reader,
)
from pypic.readers._simple import SimpleReader, open_simple
from pypic.readers.base import (
    FieldDataset,
    GridInfo,
    SimulationConfig,
    SimulationReader,
)
from pypic.readers.batsrus import (
    BATSRUSConfig,
    BATSRUSOutputFormat,
    BATSRUSReader,
    open_batsrus,
    parse_param_in,
)
from pypic.readers.config import load_config
from pypic.readers.ipic3d import (
    ConservedQuantities,
    IPic3DConfig,
    IPic3DH5hutReader,
    IPic3DParallelReader,
    IPic3DSerialReader,
    load_conserved_quantities,
    open_ipic3d,
    parse_inp,
)
from pypic.readers.openggcm import (
    OpenGGCMGrid,
    OpenGGCMReader,
    open_openggcm,
    parse_grid_file,
)

__all__ = [
    "BATSRUSConfig",
    "BATSRUSOutputFormat",
    "BATSRUSReader",
    "ConservedQuantities",
    "FieldDataset",
    "GridInfo",
    "IPic3DConfig",
    "IPic3DH5hutReader",
    "IPic3DParallelReader",
    "IPic3DSerialReader",
    "OpenGGCMGrid",
    "OpenGGCMReader",
    "ReaderEntry",
    "SimpleReader",
    "SimulationConfig",
    "SimulationReader",
    "load_config",
    "load_conserved_quantities",
    "open_batsrus",
    "open_ipic3d",
    "open_openggcm",
    "open_simple",
    "open_simulation",
    "parse_grid_file",
    "parse_inp",
    "parse_param_in",
    "register_reader",
    "registered_readers",
    "unregister_reader",
]
