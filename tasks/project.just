# Run all core project tasks in sequence
[group('project')]
project: project-requirements (project-run "tuning") (project-run "training") (project-run "evaluations") (project-run "inference")

# Export environment configuration for deployment/Docker
[group('project')]
project-environment: project-requirements
    #!/usr/bin/env python3
    import json
    from pathlib import Path

    python_ver = Path(".python-version").read_text().strip()

    with open("requirements.txt", "r") as f:
        dependencies = [
            line.split("==")[0].strip()
            for line in f
            if line.strip() and not line.startswith(("#", "pywin32", "-e"))
        ]

    configuration = {"python": python_ver, "dependencies": dependencies}
    with open("python_env.yaml", "w") as f:
        json.dump(configuration, f, indent=4)
        f.write("\n")

# Export uv dependencies to requirements.txt
[group('project')]
project-requirements:
    uv export --format=requirements-txt --no-dev --no-hashes \
        --no-editable --no-emit-project --output-file=requirements.txt

# Run a specific job using MLflow Project entry points
# Usage: just project-run training
[group('project')]
project-run job:
    @echo "Executing Job: {{job}}..."
    PYTHONPATH=src uv run mlflow run . \
        --env-manager local \
        --experiment-name={{REPOSITORY}} \
        --run-name={{capitalize(job)}} \
        -P conf_file=confs/{{job}}.yaml

# Specialized command to run the full pipeline with one command
[group('project')]
pipeline:
    just project-run tuning
    just project-run training
    just project-run evaluations
    just project-run promotion
    just project-run inference
    just project-run explanation

# Clean up temporary logs and local parquet results
[group('project')]
project-clean:
    rm -rf data/predictions/*.parquet
    rm -rf outputs/plots/*.png
