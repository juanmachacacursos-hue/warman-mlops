# Imagen de la interfaz: solo instala Gradio y httpx (no el modelo).
# Se construye desde la raíz del repositorio:  docker build -f app/Dockerfile .
FROM python:3.13.9-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group ui --no-install-project

COPY app/gradio_app.py ./app/gradio_app.py

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["python", "app/gradio_app.py"]
