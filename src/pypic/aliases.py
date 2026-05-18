"""Public name-aliases for codegen consumers.

Cross-tool codegen pipelines (webpic, rustpic tooling) read these
tables to generate TypeScript/Rust mirrors of pypic's name resolution.
Stable surface — additions only, no renames.
"""

from __future__ import annotations

from pypic._aliases import (
    COMPUTE_ALIASES,
    GROUP_ALIASES,
    SPECIES_SUFFIX_RE,
    species_name_aliases,
)

__all__ = [
    "COMPUTE_ALIASES",
    "GROUP_ALIASES",
    "SPECIES_SUFFIX_RE",
    "species_name_aliases",
]
