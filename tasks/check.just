# tasks/check.just

[group('check')]
check: check-code check-type check-format check-security

# 1. Linting (Ruff)
# Checks for bugs, unused imports, and bad coding style.
[group('check')]
check-code:
    uv run ruff check {{SOURCES}} {{TESTS}}

# 2. Type Checking (Mypy)
# Verifies that your "-> int" and "X: pd.DataFrame" hints are correct.
[group('check')]
check-type:
    uv run mypy {{SOURCES}} --ignore-missing-imports

# 3. Format Checking (Ruff)
# Checks if code is perfectly aligned. Unlike 'just format', this only COMPLAINS, it doesn't fix.
[group('check')]
check-format:
    uv run ruff format --check {{SOURCES}} {{TESTS}}

# 4. Security Scanning (Bandit)
# Looks for passwords in code, SQL injection risks, or weak math functions.
[group('check')]
check-security:
    uv run bandit --recursive --configfile=pyproject.toml {{SOURCES}}

# 5. Testing & Coverage (Pytest)
# Run tests quickly without checking coverage %
[group('check')]
check-test numprocesses="auto":
    uv run pytest --numprocesses={{numprocesses}} {{TESTS}}
