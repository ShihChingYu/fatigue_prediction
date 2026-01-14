# tasks/format.just

# Main command: Runs import sorting then code formatting
[group('format')]
format: format-import format-source

# 1. Format Imports (Sorts them alphabetically and removes unused ones)
[group('format')]
format-import:
    uv run ruff check --select=I --fix {{SOURCES}} {{TESTS}}

# 2. Format Source Code (Fixes indentation, spacing, quotes)
[group('format')]
format-source:
    uv run ruff format {{SOURCES}} {{TESTS}}
