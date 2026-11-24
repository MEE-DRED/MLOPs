# One image, two roles:
#   * docker compose:   api service and dashboard service override CMD (see docker-compose.yml)
#   * Hugging Face Space / `docker run`: default CMD starts BOTH (API internal, dashboard on $PORT)
# NOT built in the sandbox that produced it (no Docker available) - see docs/deployment.md for the verification checklist.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TF_CPP_MIN_LOG_LEVEL=2

# Spaces run the container as UID 1000, so create that user and keep everything it writes inside its home
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements-docker.txt .
RUN pip install --user -r requirements-docker.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user scripts ./scripts
COPY --chown=user outputs/tables ./outputs/tables
COPY --chown=user models ./models
COPY --chown=user data ./data

# mutable state (database, uploads, retrained model versions) lives here; mount a volume on it to persist
ENV DR_STATE_DIR=/home/user/state
RUN mkdir -p /home/user/state

EXPOSE 7860 8000 8501
HEALTHCHECK --interval=30s --timeout=6s --start-period=120s --retries=3 CMD ["python", "-m", "scripts.healthcheck"]
CMD ["bash", "scripts/start_space.sh"]
