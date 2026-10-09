FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY data/wheelhouse /wheelhouse
COPY pyproject.toml requirements.txt requirements-trainer.txt ./

RUN pip install --no-index --find-links=/wheelhouse \
      "torch==2.7.1+cu126" \
      "torchvision==0.22.1+cu126" \
      -r requirements.txt \
      -r requirements-trainer.txt

COPY README.md ./
COPY src ./src
RUN pip install --no-index --find-links=/wheelhouse --no-deps .

EXPOSE 8000
CMD ["uvicorn", "agentforge.gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
