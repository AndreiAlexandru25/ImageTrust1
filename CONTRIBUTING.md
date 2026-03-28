# Contributing to ImageTrust

Thank you for your interest in contributing to ImageTrust! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Release Process](#release-process)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone regardless of experience level.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) NVIDIA GPU with CUDA for faster inference

### Finding Issues

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are open for contributions
- Feel free to ask questions on any issue before starting work

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ImageTrust.git
cd imagetrust
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# or: .venv\Scripts\activate  # Windows
```

### 3. Install Development Dependencies

```bash
# Install in editable mode with dev dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests
python -m pytest tests/unit -v

# Run linting
python -m ruff check src/ tests/
python -m black --check src/ tests/
```

---

## Code Style

We use automated tools to ensure consistent code style:

### Formatting

- **Black** for code formatting (line length: 100)
- **isort** for import sorting (black profile)

```bash
# Format code
python -m black src/ tests/
python -m isort src/ tests/
```

### Linting

- **Ruff** for fast Python linting
- **MyPy** for optional type checking

```bash
# Lint code
python -m ruff check src/ tests/

# Type check (optional)
python -m mypy src/imagetrust --ignore-missing-imports
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Run all hooks manually
pre-commit run --all-files

# Skip hooks (not recommended)
git commit --no-verify
```

### Style Guidelines

1. **Docstrings**: Use Google-style docstrings
   ```python
   def function(arg1: str, arg2: int) -> bool:
       """Short description.

       Longer description if needed.

       Args:
           arg1: Description of arg1.
           arg2: Description of arg2.

       Returns:
           Description of return value.

       Raises:
           ValueError: When something is wrong.
       """
   ```

2. **Type Hints**: Add type hints for function signatures
   ```python
   def process_image(path: str | Path) -> Dict[str, Any]:
       ...
   ```

3. **Imports**: Group imports (standard library, third-party, local)
   ```python
   import os
   from pathlib import Path

   import numpy as np
   from PIL import Image

   from imagetrust.detection import AIDetector
   ```

---

## Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit -v

# Integration tests
python -m pytest tests/integration -v -m integration

# With coverage
python -m pytest tests/ --cov=imagetrust --cov-report=html

# Skip slow tests
python -m pytest tests/ -v -m "not slow"
```

### Writing Tests

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use descriptive test names: `test_detect_returns_probability`
4. Use fixtures for common setup

```python
import pytest
from imagetrust.detection import AIDetector

@pytest.fixture
def detector():
    return AIDetector()

def test_detect_returns_valid_probability(detector, sample_image):
    result = detector.detect(sample_image)
    assert 0 <= result["ai_probability"] <= 1
```

### Test Markers

```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.integration   # Integration tests
@pytest.mark.gpu           # Requires GPU
```

---

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3. Commit Changes

We use conventional commits:

```bash
git commit -m "feat: add new detection model"
git commit -m "fix: correct probability calibration"
git commit -m "docs: update API documentation"
git commit -m "test: add integration tests for batch processing"
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what and why
- Link to related issue(s)
- Screenshots if UI changes

### 5. Code Review

- Address reviewer feedback
- Keep PR focused on single concern
- Rebase if needed to resolve conflicts

---

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions builds and publishes release

---

## Project Structure

```
imagetrust/
├── src/imagetrust/      # Main package
│   ├── api/             # FastAPI REST API
│   ├── baselines/       # Baseline detectors
│   ├── cli.py           # CLI commands
│   ├── detection/       # Core detection
│   └── ...
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── examples/            # Example scripts
├── scripts/             # Utility scripts
├── docs/                # Documentation
└── configs/             # Configuration files
```

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email maintainers directly

---

## Recognition

Contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md for significant contributions
- Release notes

Thank you for contributing to ImageTrust!
