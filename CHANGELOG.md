# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.4] - 2024-08-28

### Added

- Function `array_of_tuples`.
- Export statement for `local_permutation`.
- Experimental support for block arrays via types `BRange`, `BVector`, `BMatrix`, and `BArray`.

## [0.5.3] - 2024-08-16

### Fixed

- Typo: `node_coorinates_unit_cube` -> `node_coordinates_unit_cube`.
- Bug in `nullspace_linear_elasticity`.
- Bug in `PVector` when working in split format.

### Deprecated

- `near_nullspace_linear_elasticity` in favor of `nullspace_linear_elasticity`.

## [0.5.2] - 2024-08-13

### Added

- Split format support for `PVector`.
- Helper functions to build partitioned sparse matrices and vectors in split format, `pvector_from_split_blocks` and `psparse_from_split_blocks`.
- Gallery function `linear_elasticity_fem`.
- Function `global_to_owner`.

### Deprecated

- `OwnAndGhostVectors` is replaced by `SplitVector`.

## [0.5.1] - 2024-07-26

### Added

- Function `spmv!`.

### Fixed

- Performance improvements in sparse matrix-vector multiplication.

## [0.5.0] - 2024-07-26

### Changed

- `MPIArray` and `DebugArray` are now inmutable for performance reasons.

### Fixed

-  Many type instabilities.
-  `mul!` (and also `consistent!`, `exchange!` ) allocation free for `MPIArray` (with `check-bounds=no`)

### Removed

- `OldPSparseMatrix` and related functionality.

### Added

- New method for `psparse` that accepts a sparse matrix type as first argument.

## [0.4.7] - 2024-07-18

### Added

- Functions `laplacian_fdm` and `laplacian_fem` to generate distributed sparse matrices from the discretization of the Laplacian operator.

## [0.4.6] - 2024-05-25

### Fixed

- Bug in `consistent!` for sparse matrices.

## [0.4.5] - 2024-05-17

### Fixed

- Bug in `copy`.
- Bug in sparse matrix-matrix products.
- Performance improvements in `tuple_of_arrays`.

### Added

- Function `centralize` for sparse matrix.
- `multicast` for arbitrary types.

## [0.4.4] - 2024-02-20

### Fixed

- Bug in `psparse`.

### Added

- Distributed sparse matrix-matrix multiplication routines.
- Distributed transposed sparse matrix-vector product.

## [0.4.3] - 2024-02-09

### Added

- Function `sparse_matrix`, which is is equivalent to `sparse`, but it allows one to pass negative indices (which will be ignored). Useful to handle boundary conditions.

## [0.4.2] - 2024-02-07

### Added

- Enhancements in the logic behind the (undocumented) optional arguments in `psparse`, `pvector`, and `psystem`.

## [0.4.1] - 2024-02-05

### Added

- Gather/scatter for non isbitstype objects.
- Function `find_local_indices`.

## [0.4.0] - 2024-01-21

### Changed

- Major refactoring in `PSparseMatrix` (and in `PVector` in a lesser extent).
The old code is still available (but deprecated), and can be recovered applying this renaming to your code-base:
  - `PSparseMatrix -> OldPSparseMatrix`
  - `psparse! -> old_psparse!`
  - `pvector! -> old_pvector!`
  - `trivial_partition -> old_trivial_partition`

- The default parallel sparse matrix format is now split into 4 blocks corresponding to own/ghost columns/rows.
The previous "monolithic" storage is not implemented anymore for the new version of `PSparseMatrix`, but can be implemented in the new setup if needed.
- `emit` renamed to `multicast`. The former name is still available but deprecated.

### Added

- Efficient re-construction of `PSparseMatrix` and `PVector` objects.
- Functions `assemble` and `consistent` (allocating versions of `assemble!` and `consistent!` with a slightly different
treatment of the ghost rows).
- Function `consistent` for `PSparseMatrix`.
- Functions `repartition` and `repartition!` used to change the data partition of `PSparseMatrix` and `PVector` objects.
- Functions `psystem` and `psystem!` for generating a system matrix and vector at once.
- Function `trivial_partition`.
- Support for sub-assembled matrices in `PSparseMatrix`.


## [0.3.4] - 2023-09-06

### Added 

- Function `partition_from_color`.
- `Base.copyto!` and `Base.copy!` for `PSparseMatrix`.

### Fixed

- Bugfix: `Base.similar` methods for `PSparseMatrix` not working.

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

