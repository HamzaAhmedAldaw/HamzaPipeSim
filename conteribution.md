
1. **AI-Generated Code**: All code must include AI generation markers
   ```cpp
   /// AI_GENERATED: Description
   /// Generated on: YYYY-MM-DD
   ```

2. **Testing**: All new features must include unit tests
   - C++ tests use GoogleTest
   - Python tests use pytest

3. **Documentation**: Update documentation for any API changes

4. **Code Style**:
   - C++: Follow Google C++ Style Guide
   - Python: Follow PEP 8
   - Use provided formatting tools

### Building from Source

```bash
# Clone repository
git clone https://github.com/pipeline-sim/pipeline-sim.git
cd pipeline-sim

# Build C++ core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Install Python package in development mode
cd ../python
pip install -e .[dev]

# Run Python tests
pytest tests/
```

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Example:
```
Add Beggs-Brill correlation implementation

- Implement two-phase flow correlation
- Add unit tests for all flow regimes
- Update documentation with theory

Fixes #123
```

## Project Structure

- `core/`: C++ simulation engine
- `python/`: Python bindings and high-level API
- `plugins/`: Extension modules
- `docs/`: Documentation source
- `tests/`: Integration tests
- `examples/`: Example scripts and notebooks

## Testing

### Running Tests

```bash
# C++ tests
cd build
ctest

# Python tests
cd python
pytest

# With coverage
pytest --cov=pipeline_sim --cov-report=html
```

### Writing Tests

- Test files should be named `test_*.cpp` or `test_*.py`
- Each test should be independent
- Use meaningful test names that describe what is being tested
- Include edge cases and error conditions

## Documentation

- API documentation uses Doxygen (C++) and Sphinx (Python)
- Include docstrings for all public functions
- Provide examples in documentation
- Keep README.md updated

## Release Process

1. Update version numbers in:
   - `CMakeLists.txt`
   - `python/setup.py`
   - `core/include/pipeline_sim/pipeline_sim.h`

2. Update CHANGELOG.md
3. Create release tag
4. Build and test on all platforms
5. Deploy to package repositories

## Questions?

Feel free to open an issue for any questions about contributing!
EOF

# Create initial source files