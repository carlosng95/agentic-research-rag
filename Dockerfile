FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY agentic_research_rag ./agentic_research_rag

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "torch==2.14.0+cpu" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/papers /app/data/indexes /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agentic_research_rag.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]