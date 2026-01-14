# MLflow management tasks
[group('mlflow')]
mlflow: mlflow-serve

# Start the MLflow UI/Server
[group('mlflow')]
mlflow-serve host="127.0.0.1" port="5000" uri="./mlruns":
    @echo "Starting MLflow server at http://{{host}}:{{port}}..."
    uv run mlflow server --host={{host}} --port={{port}} --backend-store-uri={{uri}}

# Run mlflow doctor to check the environment configuration
[group('mlflow')]
mlflow-doctor:
    uv run mlflow doctor

# Clean up any uncommitted/temporary MLflow data (use with caution)
[group('mlflow')]
mlflow-gc uri="./mlruns":
    @echo "Running garbage collection on MLflow store..."
    uv run mlflow gc --backend-store-uri={{uri}}

# List all experiments currently in the tracking server
[group('mlflow')]
mlflow-experiments:
    uv run mlflow experiments search
