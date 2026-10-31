# Imagen de la API: instala dependencias, genera datos, entrena (con puerta de
# calidad) y sirve el modelo con FastAPI.
FROM python:3.13.9-slim-bookworm

# uv fijado a una versión exacta
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Capa 1: dependencias (se reutiliza mientras no cambien pyproject.toml / uv.lock)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project

# Capa 2: código del proyecto
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups

# Capa 3: datos + entrenamiento. Si el modelo no pasa la puerta de calidad,
# `entrenar` termina con error y la imagen NO se construye.
RUN preparar-datos && entrenar && rm -rf data/raw

# No correr como root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\",\"8000\")}/health')"

CMD ["sh", "-c", "uvicorn warman.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
