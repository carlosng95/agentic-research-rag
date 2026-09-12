FROM python:3.11-slim AS dependencies

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m venv $VIRTUAL_ENV

COPY pyproject.toml /tmp/pyproject.toml

RUN python - <<'PY'
import tomllib
from pathlib import Path

with open("/tmp/pyproject.toml", "rb") as file:
    pyproject = tomllib.load(file)

dependencies = pyproject["project"]["dependencies"]

Path("/tmp/requirements.txt").write_text(
    "\n".join(dependencies),
    encoding = "utf-8",
)
PY

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "torch==2.14.0+cpu" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/indexes /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

COPY --from=dependencies /opt/venv /opt/venv

COPY --chown=appuser:appuser agentic_research_rag ./agentic_research_rag

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agentic_research_rag.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]