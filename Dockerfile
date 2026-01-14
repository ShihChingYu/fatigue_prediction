# https://docs.docker.com/engine/reference/builder/

FROM ghcr.io/astral-sh/uv:python3.9-bookworm

# Copy the wheel file from your local dist/ folder
COPY dist/*.whl .

# Install the package globally in the container
RUN uv pip install --system *.whl

# Default command
CMD ["fatigue", "--help"]
