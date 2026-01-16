# Contributing to SotaVideoRAG

Thank you for your interest in contributing to SotaVideoRAG! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/SotaVideoRAG.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit with clear messages: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest black flake8 isort

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Code Style

We follow PEP 8 with some modifications:

- Maximum line length: 127 characters
- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Use flake8 for linting: `flake8 .`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=./ --cov-report=html

# Run specific test file
pytest tests/test_videorag.py
```

## Submitting Changes

### Pull Request Process

1. Update documentation if you're adding new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Reference any related issues in your PR description

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts

## Areas for Contribution

### High Priority
- GPU-accelerated FAISS indexing (IVF, HNSW)
- Video streaming support
- Batch processing API
- Multi-video search capabilities

### Medium Priority
- UI/UX improvements
- Additional video format support
- Performance optimizations
- Better error handling

### Low Priority
- Documentation improvements
- Example notebooks
- Tutorial videos
- Translations

## Bug Reports

When filing a bug report, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Exact steps to trigger the bug
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - GPU/CUDA version
   - Package versions (`pip freeze`)
6. **Logs**: Relevant error messages and logs

## Feature Requests

For feature requests, please:

1. Check existing issues to avoid duplicates
2. Describe the feature and its benefits
3. Provide use cases
4. Suggest implementation approach (optional)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open a [GitHub Discussion](https://github.com/your-repo/discussions)
- Check existing issues
- Read the documentation

Thank you for contributing!
