# Contributing to The Intelligence Bound

Thank you for your interest in contributing to The Intelligence Bound project! This document provides guidelines for contributions.

## Types of Contributions

### 1. Bug Reports

If you find a bug in the code or a numerical error in the calculations, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your Python version and operating system
- Relevant code snippets or error messages

### 2. Scientific Feedback

We welcome scientific discussion and critique:

- **Theoretical critiques**: Issues with the derivations or assumptions
- **Empirical validation**: Experimental tests of the falsifiable predictions
- **Alternative models**: Competing frameworks or extensions

Please open an issue with the label `scientific-discussion`.

### 3. Code Improvements

We accept pull requests for:

- Bug fixes
- Performance improvements
- Additional D estimators
- Documentation improvements
- Test coverage expansion

### 4. Documentation

Improvements to documentation are always welcome:

- Clarifying explanations
- Adding examples
- Fixing typos
- Translating documentation

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/intelligence-bound.git
   cd intelligence-bound
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Standards

### Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function signatures
- Include docstrings with examples for public functions
- Format code with `black`

### Testing

- All new code must include tests
- Maintain or improve test coverage
- Run the full test suite before submitting:
  ```bash
  pytest tests/ -v
  ```

### Documentation

- Update docstrings for any modified functions
- Add examples where helpful
- Update README.md if adding new features

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation as needed
3. Add a clear description of the changes
4. Reference any related issues
5. Wait for review from maintainers

## Scientific Integrity

When contributing to the scientific content:

- All claims must be mathematically derivable or empirically testable
- Cite sources for any new data or estimates
- Clearly distinguish between established results and speculation
- Preserve the falsifiability of the theory

## Code of Conduct

- Be respectful and constructive
- Focus on the work, not the person
- Welcome diverse perspectives and backgrounds
- Assume good faith in discussions

## Questions?

Open an issue with the label `question` or contact the maintainer at viridisnorthllc@gmail.com.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
