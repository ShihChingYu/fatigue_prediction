# tasks/commit.just

# bump package version based on commits (e.g. 0.1.0 -> 0.2.0)
[group('commit')]
commit-bump:
    uv run cz bump

# interactive wizard to create a standardized commit
[group('commit')]
commit-files:
    uv run cz commit

# get info about the commitizen environment
[group('commit')]
commit-info:
    uv run cz info
