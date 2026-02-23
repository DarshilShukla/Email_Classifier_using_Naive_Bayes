# Contributing to SMS Spam Classifier

Thank you for your interest in contributing to the SMS Spam Classifier project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for any new functionality
4. **Ensure all tests pass** by running `pytest`
5. **Update documentation** if needed
6. **Commit your changes** with clear, descriptive commit messages
7. **Push to your fork** and submit a pull request

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Email_Classifier_using_Naive_Bayes.git
   cd Email_Classifier_using_Naive_Bayes
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python setup_nltk.py
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Comment complex logic

## Testing Guidelines

- Write unit tests for new functions
- Write integration tests for new features
- Ensure test coverage remains above 90%
- Use property-based testing for universal properties
- Test edge cases and error conditions

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose. The review process includes:

1. Automated tests must pass
2. Code style must be consistent
3. Documentation must be updated
4. At least one maintainer approval required

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

Thank you for contributing! 🎉
