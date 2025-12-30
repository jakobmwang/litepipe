# Slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies via uv
# We use --system to install into the environment provided by the base image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Copy the source code folder keeping the structure
# Result: /app/src/main.py exists
COPY src ./src

EXPOSE 5000

# We use src.main because the file is located at /app/src/main.py
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]