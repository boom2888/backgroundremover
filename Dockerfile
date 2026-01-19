FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git needed for DVC)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies including dvc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install dvc

# Copy your source code, configs, and artifacts (model files)
COPY src/ src/
COPY configs/ configs/
COPY artifacts/ artifacts/

# Copy scripts and root python files if any
COPY app.py .
COPY template.py .
COPY test.py .

# Initialize git for DVC
RUN git init && \
    git config user.email "docker@build.local" && \
    git config user.name "Docker Build"

# Configure DVC remote (you need to provide these args at build time)
ARG DAGSHUB_USER
ARG DAGSHUB_PASSWORD

RUN dvc remote list | grep -q dagshub || dvc remote add -d dagshub https://dagshub.com/"${DAGSHUB_USER}"/"{REPO}".dvc
RUN dvc remote modify dagshub auth basic

RUN dvc remote modify dagshub --local user "${DAGSHUB_USER}" && \
    dvc remote modify dagshub --local password "${DAGSHUB_PASSWORD}" && \
    dvc status checkpoint.pth.dvc && \
    dvc pull checkpoint.pth.dvc -v

# Expose your app port (adjust if different)
EXPOSE 8000

# Start FastAPI server (adjust if app location is different)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
