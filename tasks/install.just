# tasks/install.just

# Main command: Installs dependencies and sets up git hooks
[group('install')]
install: install-project install-hooks

# 1. Install Python dependencies
[group('install')]
install-project:
    # --all-groups ensures dev tools like pytest/black are included.
    uv sync --all-groups

# 2. Install Git Hooks (Auto-checking code before you commit)
[group('install')]
install-hooks:
    uv run pre-commit install --hook-type=pre-push
    uv run pre-commit install --hook-type=commit-msg
