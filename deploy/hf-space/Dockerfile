# FedRAG API as a Hugging Face Space (Docker SDK), the public demo by default.
#
# Build context: the folder scripts/build_hf_space.sh assembles (deploy/build/hf-space), laid out
# like the repository so the backend's default paths work unchanged:
#   backend/app, backend/constraints.txt, backend/requirements.txt (from pyproject.toml)
#   data/eval/*.json, data/eval/stance_gold.jsonl, data/index/{chunks.jsonl,manifest.json}
# No secrets are copied in: DATABASE_URL is a Space secret, read at run time.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Dependencies first, so editing the app or the data does not reinstall them. torch comes from
# PyTorch's CPU index (Spaces' free hardware has no GPU, and the CUDA wheels are gigabytes), at
# the version constraints.txt pins; everything else from PyPI at the pinned versions.
WORKDIR /tmp/deps
COPY backend/constraints.txt backend/requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep -E '^torch==' constraints.txt)" \
    && pip install -r requirements.txt -c constraints.txt \
    && rm -rf /tmp/deps

# Spaces run the container as user 1000: create it and give it the app.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user backend/app backend/app
COPY --chown=user data data

# The public demo: no OpenAI key needed and none used (backend/app/api/demo.py). Space variables
# override these; DATABASE_URL (a read-only role, deploy/neon_readonly.sql) is a Space secret.
ENV PUBLIC_DEMO=true \
    ENVIRONMENT=production \
    PORT=7860

WORKDIR /home/user/app/backend
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/api/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn --factory app.main:create_app --host 0.0.0.0 --port \"${PORT:-7860}\""]
