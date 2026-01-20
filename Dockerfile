# https://docs.docker.com/engine/reference/builder/

FROM ghcr.io/astral-sh/uv:python3.9-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

COPY src/ ./
COPY main.py .
COPY python_model.pkl .

EXPOSE 5001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
