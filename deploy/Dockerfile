# The pickspin image: the package with the [live] and [classifier] extras on CPU-only torch, the
# fine-tuned DistilBERT classifier, the benchmark queries and the deploy files. It runs the gateway
# (pickspin serve), the load test (pickspin replay) and the benchmark replay (pickspin live), all on CPU;
# the model servers are separate vLLM containers.
#
# .github/workflows/image.yml builds it and pushes ghcr.io/bhanuprakashvangala/pick-and-spin. To build it
# yourself, from the repository root:
#   docker build -f deploy/Dockerfile -t pick-and-spin .
# The entrypoint is the pickspin command, run in /app, which holds data/, deploy/ and models/, so its
# default paths resolve as in a checkout.
FROM python:3.11-slim
WORKDIR /app
# CPU-only torch first, in its own layer, so that the [classifier] extra does not pull a CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[live,classifier]"
# The classifier comes from the Hugging Face Hub, so a build needs no local copy of it.
ARG CLASSIFIER=bhanuprakashvangala/pickspin-distilbert-complexity
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${CLASSIFIER}', local_dir='models/distilbert-complexity-classifier')" \
    && rm -rf models/distilbert-complexity-classifier/.cache
COPY data ./data
COPY deploy ./deploy
ENTRYPOINT ["pickspin"]
