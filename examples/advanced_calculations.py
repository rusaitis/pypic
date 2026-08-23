"""Advanced derived quantity examples beyond the built-in compute() chain.

These snippets show how to handle cases that the automatic resolution
doesn't cover — multi-species sums, custom derived fields, and manual
tensor operations.
"""

# %% Total pressure from N species
# compute("P") handles the common 2-species case (Pe + Pi) automatically.
# For 3+ species, sum explicitly:

# n_species = len(sim.species)
# p_total = sum(data.compute(f"P_s{i}") for i in range(n_species))
# data = data.with_field("P", p_total, "pressure")

# %% Total current density from per-species currents
# J = sum(J_s) is usually available directly, but if not:

# for c in ("1", "2", "3"):
#     j_total = sum(data[f"J{c}_s{i}"] for i in range(n_species))
#     data = data.with_field(f"J{c}", j_total, "current_density")

# %% Custom derived field with metadata
# Attach any array as a named field for plotting and unit conversion:

# import numpy as np
# e_ratio = data.compute("e_B") / (data.compute("e_k") + 1e-30)
# data = data.with_field("e_ratio", e_ratio, "dimensionless",
#                        long_name="Magnetic-to-kinetic energy ratio")

# %% Per-species plasma beta for species index > 1
# beta_e and beta_i are built in. For species 2+, use the template:

# beta_s2 = data.compute("beta_s2")  # auto-resolves P_s2 and |B|

# %% Selective field loading with dependency resolution
# read(fields=...) expands shorthand and resolves compute dependencies:
#
#   fields=["B"]       → B1, B2, B3
#   fields=["J_s0"]    → J1_s0, J2_s0, J3_s0
#   fields=["Pi"]      → P11_s1..P33_s1 (full tensor)
#   fields=["beta"]    → P11_s0..P33_s1, B1, B2, B3
#   fields=["EFi"]     → EF1_s1, EF2_s1, EF3_s1
