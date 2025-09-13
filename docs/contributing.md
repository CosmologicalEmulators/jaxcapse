# Contributing

We welcome contributions to jaxcapse! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/jaxcapse.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Create a new branch: `git checkout -b feature-name`

## Testing

Run tests before submitting:
```bash
pytest tests/
```

## Documentation

Build documentation locally:
```bash
mkdocs build
mkdocs serve  # Preview at http://localhost:8000
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Submit pull request to `develop` branch