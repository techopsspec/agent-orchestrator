# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12-slim
FROM docker.io/library/python:$PYTHON_VERSION AS base

WORKDIR /app

# curl for the compose healthcheck, matching Operator-Portal's own backend Dockerfile.
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y curl && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

FROM base AS build
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM base
COPY --from=build /install /usr/local
COPY src ./src

EXPOSE 8100
CMD ["uvicorn", "agent_orchestrator.main:app", "--host", "0.0.0.0", "--port", "8100"]
