# StrikeSim Test Suite

This directory contains the test suite for the StrikeSim project using pytest.

## Test Structure

- `test_basic_functionality.py` - Basic functionality tests for core simulation features
- `test_network_robustness.py` - Tests for network robustness and edge cases
- `test_dashboard_visualization.py` - Tests for network visualization functionality

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_basic_functionality.py
```

### Run specific test function:
```bash
pytest tests/test_basic_functionality.py::test_basic_simulation
```

### Run tests with verbose output:
```bash
pytest -v
```

### Run tests and show coverage:
```bash
pytest --cov=strikesim
```

### Run only fast tests (exclude slow ones):
```bash
pytest -m "not slow"
```

## Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test how components work together
- **Robustness Tests**: Test edge cases and error conditions
- **Visualization Tests**: Test plotting and visualization functions

## Adding New Tests

1. Create a new test file with the prefix `test_`
2. Use descriptive test function names starting with `test_`
3. Use pytest fixtures for common setup
4. Add appropriate assertions and error messages
5. Mark slow tests with `@pytest.mark.slow`
6. Mark integration tests with `@pytest.mark.integration`

## Test Data

Test data files (CSV, GEXF) are created temporarily during tests and cleaned up automatically. 