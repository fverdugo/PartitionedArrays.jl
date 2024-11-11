
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2024-11-11

### Added

- Function `default_solver`
- First draft of a Newton-Raphson solver in function `newton_raphson`.

## [0.3.0] - 2024-10-29

### Changed

- Linear solver interface.

### Added

- Interface for parallel nonlinear and ode solvers

## [0.2.2] - 2024-10-03

### Added

- Support for user-defined near nullspace in AMG solver.
- Flag `history` to return solve history in `solve!`.
- Expose iterations for iterative solvers with function `iterations!`.
- Solver traits `uses_nullspace` and `uses_initial_guess`. These allow the user to build the nullspace or reset the initial guess to zero only when needed by the solver.

## [0.2.1] - 2024-07-26

### Added
- Support for PartitionedArrays v0.5.

## [0.2.0]

### Changed

- Linear solver interface.

## [0.1.0]

### Added

- Basic smoothed aggregation algebraic multigrid preconditioner for the Laplace problem
