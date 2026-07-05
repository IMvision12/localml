# InferML - local web server with an OpenAI-compatible API.
#
# Two stages: build the self-contained frontend with Node, then install the
# Python package (server + inference extra) into a slim runtime image. No
# Electron, no native binary.
#
#   docker build -t inferml .
#   docker run --rm -p 11500:11500 inferml
#   # GPU: docker run --rm --gpus all -p 11500:11500 inferml
#
# Model weights download to the HF cache at runtime; mount a volume to persist:
#   docker run --rm -p 11500:11500 -v inferml-hf:/root/.cache/huggingface inferml

# ---- stage 1: build the frontend ----
FROM node:20-slim AS webui
WORKDIR /app
COPY package.json package-lock.json ./
# Only the deps needed to build + vendor the renderer (no electron/sharp).
RUN npm install --no-audit --no-fund esbuild react react-dom marked dompurify
COPY scripts ./scripts
COPY src ./src
COPY python/supported_architectures.json ./python/supported_architectures.json
RUN node scripts/build-renderer.js && node scripts/bundle-webui.js

# ---- stage 2: python runtime ----
FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    INFERML_DATA_DIR=/data

COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY python ./python
# Bring in the frontend + routing data bundled by stage 1.
COPY --from=webui /app/python/server/webui ./python/server/webui
COPY --from=webui /app/python/server/_data ./python/server/_data

# Install the server + the full inference stack (CPU wheels by default). For a
# CUDA image, base this stage on an nvidia/cuda image and add the cu124 index.
RUN pip install ".[inference]"

EXPOSE 11500
VOLUME ["/data"]
CMD ["inferml", "--host", "0.0.0.0", "--no-browser"]
