# Testing Guide

## Running Tests

### Quick Start

```bash
# Run all tests
poetry run pytest tests/

# Run with coverage
poetry run pytest tests/ --cov=jaxcapse

# Run specific test file
poetry run pytest tests/test_core_functionality.py
```

### Test Coverage

Current coverage: **100%** (47/47 lines)

```bash
# Generate coverage report
poetry run pytest tests/ --cov=jaxcapse --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Structure

### Test Files

| File | Purpose | Tests |
|------|---------|-------|
| `test_core_functionality.py` | Core MLP functionality | 13 |
| `test_inference.py` | Inference and batching | 15 |
| `test_edge_cases.py` | Error handling | 13 |
| `test_jax_features.py` | JAX transformations | 12 |
| `test_utils.py` | Utility functions | 7 |

### Test Categories

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: End-to-end workflows
3. **Edge Cases**: Boundary conditions and errors
4. **Performance Tests**: Speed and memory usage

## Writing Tests

### Test Guidelines

- Focus on software functionality, not cosmology
- Use fixtures from `tests/fixtures.py`
- Keep tests fast (< 1 second each)
- Test both success and failure cases
- Use descriptive test names

### Example Test

```python
def test_emulator_loading(mock_emulator_directory):
    """Test that emulator loads correctly."""
    emulator = jaxcapse.load_emulator(str(mock_emulator_directory))
    
    # Check correct type
    assert isinstance(emulator, jaxcapse.MLP)
    
    # Check attributes exist
    assert emulator.emulator is not None
    assert emulator.in_MinMax is not None
    assert emulator.out_MinMax is not None
```

### Using Fixtures

```python
from tests.fixtures import *

def test_with_mock_emulator(mock_emulator_directory):
    """Test using mock emulator fixture."""
    emulator = jaxcapse.load_emulator(str(mock_emulator_directory))
    # Test code here
```

## Continuous Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Python 3.10, 3.11, 3.12
- Ubuntu and macOS

### CI Workflow

```yaml
# .github/workflows/tests.yml
- Run tests with pytest
- Generate coverage report
- Upload to codecov (optional)
- Run linting checks
```

## Test Commands

### Common Commands

```bash
# Run all tests verbose
poetry run pytest tests/ -v

# Run with specific marker
poetry run pytest tests/ -m "not slow"

# Run in parallel
poetry run pytest tests/ -n auto

# Run with warnings
poetry run pytest tests/ -W error

# Debug failed test
poetry run pytest tests/ --pdb
```

### Coverage Commands

```bash
# Terminal report
poetry run pytest tests/ --cov=jaxcapse --cov-report=term-missing

# HTML report
poetry run pytest tests/ --cov=jaxcapse --cov-report=html

# XML for CI
poetry run pytest tests/ --cov=jaxcapse --cov-report=xml
```

## Debugging Tests

### Print Debugging

```python
def test_with_debugging(capsys):
    """Test with print debugging."""
    result = some_function()
    
    # Capture print output
    captured = capsys.readouterr()
    print(f"Debug: result = {result}")
    
    assert result == expected
```

### Using pdb

```python
def test_with_pdb():
    """Test with debugger."""
    import pdb; pdb.set_trace()
    # Debugger stops here
    result = complex_function()
    assert result == expected
```

### JAX Debugging

```python
# Disable JIT for debugging
import jax
jax.config.update('jax_disable_jit', True)

# Enable NaN checking
jax.config.update('jax_debug_nans', True)
```

## Performance Testing

### Timing Tests

```python
def test_performance(benchmark):
    """Test with pytest-benchmark."""
    result = benchmark(emulator.get_Cl, params)
    assert result is not None
```

### Memory Testing

```python
import tracemalloc

def test_memory_usage():
    """Test memory consumption."""
    tracemalloc.start()
    
    # Run operation
    result = process_large_batch(data)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Check memory usage
    assert peak < 1_000_000_000  # Less than 1GB
```

## Test Data

### Mock Data

Use fixtures for consistent test data:

```python
@pytest.fixture
def sample_cosmological_params():
    """Standard test parameters."""
    return np.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])
```

### Real Data

For integration tests with real emulators:

```python
def test_real_emulator():
    """Test with actual trained emulator."""
    if not Path("trained_emu/TT").exists():
        pytest.skip("Real emulator not available")
    
    emulator = jaxcapse.load_emulator("trained_emu/TT/")
    # Test with real emulator
```

## Best Practices

1. **Isolation**: Tests should not depend on each other
2. **Repeatability**: Use fixed seeds for randomness
3. **Clarity**: Clear test names and docstrings
4. **Coverage**: Aim for >85% code coverage
5. **Speed**: Keep test suite under 30 seconds

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Check PYTHONPATH and installation |
| Fixture not found | Import from `tests.fixtures` |
| Tests slow | Use pytest-xdist for parallel execution |
| Coverage low | Add tests for uncovered lines |

### Getting Help

- Check existing tests for examples
- Review pytest documentation
- Open GitHub issue for bugs