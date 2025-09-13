# JaxCapse Test Suite

## Overview
Unit tests for jaxcapse focusing on software functionality rather than cosmological validation. The tests verify the neural network emulator works correctly without making assumptions about cosmology or physical scaling.

## Test Structure

### Core Test Files

#### `fixtures.py`
Shared test fixtures and utilities including:
- Mock neural network configurations
- Sample weights and normalization parameters
- Temporary emulator directories for testing
- Edge case parameter sets

#### `test_core_functionality.py` (13 tests)
Tests core MLP functionality:
- **MLPInitialization**: Emulator initialization and attributes
- **Normalization**: JAX array handling and normalization through jaxace
- **EmulatorLoading**: File loading and path handling
- **EmulatorDescription**: Metadata and description fields

#### `test_inference.py` (15 tests)
Tests inference capabilities:
- **GetCl**: Single input inference, JIT compilation, determinism
- **BatchProcessing**: Batch inference, empty/large batches
- **InputVariations**: Different input types (numpy, JAX arrays)

#### `test_edge_cases.py` (13 tests)
Tests error handling and boundary conditions:
- **InputValidation**: Wrong dimensions, NaN/Inf handling
- **FileHandling**: Missing files, corrupted data
- **PostprocessingErrors**: Syntax errors, missing functions
- **BatchEdgeCases**: Large batches, identical inputs
- **DTypeHandling**: float32, integer inputs

## Running Tests

### Run all tests with coverage
```bash
poetry run pytest tests/ -v --cov=jaxcapse --cov-report=term-missing
```

### Run specific test file
```bash
poetry run pytest tests/test_core_functionality.py -v
```

### Run specific test class
```bash
poetry run pytest tests/test_edge_cases.py::TestFileHandling -v
```

### Run with parallel execution
```bash
poetry run pytest tests/ -n auto
```

## Test Coverage
Current coverage: **100%** (47/47 lines)

## Key Testing Principles

1. **Software Focus**: Tests verify code functionality, not cosmological accuracy
2. **No Cosmological Assumptions**: Avoid testing physical scaling or cosmological relationships
3. **JAX Arrays Required**: Tests enforce that users provide JAX arrays (no automatic list conversion)
4. **Error Handling**: Comprehensive edge case testing for robustness
5. **Performance**: Large batch tests verify memory efficiency

## CI/CD Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Python 3.10, 3.11, 3.12
- Ubuntu and macOS

See `.github/workflows/tests.yml` for CI configuration.

## Adding New Tests

When adding tests:
1. Focus on software behavior, not physics
2. Use fixtures from `fixtures.py` for consistency
3. Test both success and failure cases
4. Ensure tests are deterministic (use fixed seeds)
5. Keep tests fast (< 1 second each)

## Dependencies

Test dependencies (in `pyproject.toml`):
- pytest ^8.4
- pytest-cov ^7.0
- pytest-xdist ^3.6
- pytest-benchmark ^4.0
- hypothesis ^6.100