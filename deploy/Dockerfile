# Router image for deploy/router-job.yaml. Build from the repository root after training or
# copying the DistilBERT classifier into models/distilbert-complexity-classifier:
#   docker build -f deploy/Dockerfile -t <registry>/pick-and-spin-router .
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt transformers
COPY src ./src
COPY data ./data
COPY deploy ./deploy
COPY models ./models
