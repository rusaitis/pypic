# Security Policy

## Supported versions

pypic is early-stage research software. Fixes land on `main` and go out in the
next release; there are no long-term support branches.

| Version | Supported |
|---------|-----------|
| latest `0.1.x` release | yes |
| anything older | no — upgrade first |

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/rusaitis/pypic/security/advisories/new).
Please do not open a public issue for a suspected vulnerability.

Include what you have: affected version, a description, and a reproduction if
one exists. Expect an acknowledgement within about a week — this is a
single-maintainer scientific project, not a staffed security team.

## Scope

pypic reads simulation output written by other codes. The realistic threat is a
**malicious or malformed input file** — an HDF5, Zarr, Parquet, TOML, or
Fortran-binary file that causes a crash, unbounded memory growth, or arbitrary
code execution when parsed. Reports of that kind are in scope and welcome.

Two things are explicitly out of scope:

- **`pypic serve` exposed to an untrusted network.** It has no authentication
  and defaults to `--cors-origin "*"`. It binds to `127.0.0.1` and is intended
  for local use behind your own ingress; running it on a public interface is a
  deployment choice, not a vulnerability in pypic.
- **Vulnerabilities in dependencies**, unless pypic's own use of the dependency
  is what makes them exploitable. Report those upstream.
