FROM python:3.12-slim

# build-essential + curl/ca-certificates: safety net in case any dependency needs
# to build from source on a platform without a prebuilt wheel (e.g. some arm64
# hosts) -- most of requirements.txt (torch/onnx/onnxruntime/ezkl) ships wheels
# for linux x86_64 and aarch64, so this is a fallback, not the common path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user with UID/GID 1000 -- the common default first-user id on
# Linux and macOS -- rather than root. app/models/{zone2_circuit,proofs} and
# gateway/data are bind-mounted from the host (see docker-compose.yml, and
# scripts/tamper_suite.sh which edits those files directly by host path); files a
# root-owned container process creates there would be unwritable from the host
# afterward. Override at build time with --build-arg APP_UID=<id> if the host
# user's uid differs (check with `id -u` on Linux/macOS).
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} appuser && useradd -m -u ${APP_UID} -g ${APP_GID} appuser \
    && chown -R appuser:appuser /app \
    # pre-create the cache dirs the hf_cache/ezkl_cache named volumes mount onto,
    # correctly owned -- a brand-new named volume's mount point otherwise defaults
    # to root:root, which appuser can't write into (Docker's volume "copy-up" only
    # preserves ownership of paths that already exist in the image).
    && mkdir -p /home/appuser/.cache/huggingface /home/appuser/.ezkl \
    && chown -R appuser:appuser /home/appuser
ENV HOME=/home/appuser
USER appuser

EXPOSE 8001

# First startup downloads BAAI/bge-reranker-base (~1.1GB) from Hugging Face and
# builds the EZKL circuit for Zone 2 (settings->calibrate->compile->SRS->setup) --
# both are cached to volumes in docker-compose.yml so this only happens once.
HEALTHCHECK --interval=10s --timeout=5s --start-period=240s --retries=30 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
