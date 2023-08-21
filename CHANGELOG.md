# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added 

- Function `partition_from_color`.

## [0.3.3] - 2023-08-09

### Added 

- MPI ibarrier-based (supposedly scalable) algorithm to find rcv neighbours in a sparse all-to-all communication graph given the snd neighbors. We left the previous non-scalable algorithm as default (based on gather-scatter) until we have experimental evidence on the relative performance and scalability of the former with respect to the latter and for which core ranges.
- New kwarg `discover_cols=true` to the `psparse!` constructor, which allows the user to skip column index discovery.

### Fixed

- Bugfix: `global_length` for `PRange` not working as intended. 

## [0.3.2] - 2023-05-10

### Fixed

- Performance improvements.

## [0.3.1] - 2023-03-17

### Fixed

- Performance improvements in functions `tuple_of_arrays` and `assemble!`.

## [0.3.0] - 2023-02-01

### Changed

This version is a major refactor of the code with improvements in the software abstractions and documentation. Almost all previous functionality should be available, but with a different API.

## Previous versions

A changelog is not maintained for older versions than 0.3.0.

