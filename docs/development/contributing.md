# Contributing to JaxCapse

We welcome contributions to JaxCapse! This guide explains how to contribute.

## Development Setup

### Clone the Repository

```bash
git clone https://github.com/CosmologicalEmulators/jaxcapse.git
cd jaxcapse
```

### Install Development Dependencies

```bash
# Using Poetry (recommended)
poetry install --with dev --with docs

# Or using pip
pip install -e ".[dev,docs]"
```

### Pre-commit Hooks

Set up pre-commit hooks for code quality:

```bash
pip install pre-commit
pre-commit install
```

## Code Style

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Docstrings

Use Google-style docstrings:

```python
def function(param1: float, param2: jnp.ndarray) -> jnp.ndarray:
    """Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Example:
        >>> result = function(1.0, jnp.array([1, 2, 3]))
    """
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with coverage
poetry run pytest tests/ --cov=jaxcapse --cov-report=html

# Run specific test file
poetry run pytest tests/test_core_functionality.py

# Run in parallel
poetry run pytest tests/ -n auto
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Focus on software functionality, not cosmology
- Use fixtures from `tests/fixtures.py`

Example test:

```python
def test_emulator_loading(mock_emulator_directory):
    """Test that emulator loads correctly."""
    emulator = jaxcapse.load_emulator(str(mock_emulator_directory))
    assert isinstance(emulator, jaxcapse.MLP)
    assert emulator.emulator is not None
```

### Test Coverage

Maintain test coverage above 85%:

```bash
poetry run pytest tests/ --cov=jaxcapse --cov-report=term-missing
```

## Documentation

### Building Documentation

```bash
# Build locally
poetry run mkdocs serve

# Build static site
poetry run mkdocs build
```

### Writing Documentation

- Use Markdown for all documentation
- Include code examples
- Update API docs for new functions
- Add usage examples for new features

## Pull Request Process

### 1. Fork and Branch

```bash
# Fork repository on GitHub
git clone https://github.com/YOUR_USERNAME/jaxcapse.git
cd jaxcapse
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: description of changes"
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Type hints added
- [ ] Docstrings complete

## Issue Reporting

### Bug Reports

Include:
- JaxCapse version
- Python version
- JAX version
- Minimal reproducible example
- Error messages
- Expected behavior

### Feature Requests

Include:
- Use case description
- Proposed API
- Example code
- Alternatives considered

## Development Workflow

### Adding a New Feature

1. **Discuss**: Open an issue to discuss the feature
2. **Design**: Propose API and implementation
3. **Implement**: Write code with tests
4. **Document**: Update docs and examples
5. **Review**: Submit PR for review

### Fixing a Bug

1. **Report**: Open issue with bug details
2. **Test**: Write failing test that reproduces bug
3. **Fix**: Implement fix
4. **Verify**: Ensure test passes
5. **Submit**: Create PR with fix

## Release Process

### Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Steps

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions builds and publishes

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- Respect differing opinions
- Accept responsibility for mistakes

## Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Email: Contact maintainers directly

## Recognition

Contributors are recognized in:
- CHANGELOG.md
- GitHub contributors page
- Documentation acknowledgments

Thank you for contributing to JaxCapse!