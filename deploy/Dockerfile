# Router image for deploy/router-job.yaml: the pickspin package with the [live] and [classifier]
# extras on CPU-only torch. Build from the repository root after training or copying the DistilBERT
# classifier into models/distilbert-complexity-classifier:
#   docker build -f deploy/Dockerfile -t <registry>/pick-and-spin-router .
# The entrypoint is the pickspin command, run in /app, which holds data/, deploy/ and models/, so its
# default paths resolve as in a checkout. The Job runs `pickspin live`.
FROM python:3.11-slim
WORKDIR /app
# CPU-only torch first, in its own layer, so that the [classifier] extra does not pull a CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[live,classifier]"
COPY data ./data
COPY deploy ./deploy
COPY models ./models
ENTRYPOINT ["pickspin"]
