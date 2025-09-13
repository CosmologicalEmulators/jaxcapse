# Changelog

All notable changes to JaxCapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive unit test suite with 100% coverage
- MkDocs documentation with Material theme
- GitHub Actions CI/CD pipeline
- Support for Python 3.10, 3.11, 3.12
- Test fixtures for mock emulators
- Edge case testing
- JAX feature tests (gradients, vmap, JIT)

### Changed
- Removed backward compatibility features
- Removed wrapper methods (maximin_input, inv_maximin_output, apply)
- Enforced JAX array inputs (no automatic list conversion)
- Simplified API surface

### Fixed
- Improved error handling for missing files
- Better validation of input parameters

## [0.1.1] - 2024-01-15

### Added
- Initial public release
- Core MLP emulator class
- Support for TT, EE, TE, PP spectra
- Batch processing capability
- JAX integration (JIT, grad, vmap)
- Basic documentation

### Dependencies
- JAX >= 0.4.30
- Flax >= 0.10.0
- jaxace >= 0.1.1

## [0.1.0] - 2024-01-01

### Added
- Initial implementation
- Basic emulator loading
- Power spectrum computation

---

## Version History

### Versioning Scheme

JaxCapse follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for added functionality (backward compatible)
- **PATCH** version for backward compatible bug fixes

### Deprecation Policy

- Deprecated features are marked with warnings
- Deprecated features are removed in next MAJOR version
- Migration guides provided for breaking changes

### Support Policy

- Latest version: Full support
- Previous minor version: Security fixes only
- Older versions: No support

## Roadmap

### Planned Features

#### Version 0.2.0
- [ ] Support for massive neutrinos
- [ ] Extended ℓ range (up to 10000)
- [ ] Additional cosmological parameters

#### Version 0.3.0
- [ ] Non-linear corrections
- [ ] Cross-correlation with LSS
- [ ] Fisher matrix utilities

#### Version 1.0.0
- [ ] Stable API
- [ ] Complete documentation
- [ ] Performance optimizations
- [ ] GPU optimization

### Known Issues

- No support for non-flat cosmologies
- Limited to ΛCDM model
- Fixed neutrino masses
- No BAO/RSD emulation

### Reporting Issues

Report issues at: https://github.com/CosmologicalEmulators/jaxcapse/issues

## Contributors

- Marco Bonici (@marcobonici) - Creator and maintainer

## Acknowledgments

- jaxace developers for neural network infrastructure
- JAX team for automatic differentiation framework
- CAMB/CLASS for training data generation