FROM ghcr.io/astral-sh/uv:python3.9-bookworm

WORKDIR /app

COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ ./src/
COPY main.py .
COPY python_model.pkl* ./

ENV PYTHONPATH="/app/src:${PYTHONPATH:-}"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
