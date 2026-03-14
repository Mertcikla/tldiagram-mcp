FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files and install (cached layer)
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml --no-cache-dir --system

# Copy source
COPY main.py ./

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

CMD ["python", "main.py"]
