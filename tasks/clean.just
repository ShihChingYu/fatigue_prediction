# tasks/clean.just

# I removed 'clean-requirements' and 'clean-environment' from this list for safety
# until I have a script that auto-generates them.
[group('clean')]
clean: clean-build clean-cache clean-coverage clean-docs clean-mlruns clean-mypy clean-outputs clean-pytest clean-python clean-ruff

# clean build folders (created by 'just package')
[group('clean')]
clean-build:
    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info

# clean docs folder (created by 'mkdocs build')
[group('clean')]
clean-docs:
    rm -rf site/
    rm -rf docs/_build

# clean outputs folder (created by inference jobs)
[group('clean')]
clean-outputs:
    rm -rf outputs/*

# clean python caches (The __pycache__ folders)
[group('clean')]
clean-python:
    find . -type f -name '*.py[co]' -delete
    find . -type d -name __pycache__ -exec rm -r {} +

# clean pytest cache
[group('clean')]
clean-pytest:
    rm -rf .pytest_cache/

# clean mypy folders (created by type checking)
[group('clean')]
clean-mypy:
    rm -rf .mypy_cache/

# clean ruff cache (linter)
[group('clean')]
clean-ruff:
    rm -rf .ruff_cache/

# clean coverage files (created by 'pytest --cov')
[group('clean')]
clean-coverage:
    rm -rf .coverage
    rm -rf htmlcov/
    rm -rf coverage.xml

# clean generic cache
[group('clean')]
clean-cache:
    rm -rf .cache/

# clean mlruns folder (Warning: Deletes all local experiment history!)
[group('clean')]
clean-mlruns:
    rm -rf mlruns/
    rm -rf .mlruns/

# clean requirements file (Only run this if you auto-generate requirements.txt)
[group('clean')]
clean-requirements:
    rm -f requirements.txt

# clean environment file (Only run this if you auto-generate python_env.yaml)
[group('clean')]
clean-environment:
    rm -f python_env.yaml

# clean constraints file
[group('clean')]
clean-constraints:
    rm -f constraints.txt

# clean venv folder (Deletes entire Python environment!)
# [confirm] ensures Just asks "Are you sure?" before running this.
[confirm]
[group('clean')]
clean-venv:
    rm -rf .venv/
