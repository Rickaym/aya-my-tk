# Use Python 3.11 slim image
FROM python:3.11-slim-bullseye AS base

# Install system dependencies including Tesseract and Burmese language support
RUN apt-get update \
    && apt-get install -y tesseract-ocr tesseract-ocr-mya tesseract-ocr-eng python3-opencv

# Set working directory
WORKDIR /app

# Install UV
RUN pip install --no-cache-dir uv

# Copy the rest of the application
COPY . .

RUN uv sync --no-dev

# Create necessary directories
RUN mkdir -p data/pending

ENV GOOGLE_APPLICATION_CREDENTIALS="svc.json"
ENV PROJECT_ID="celtic-buttress-455114-i2"
ENV PROCESSOR_ID="ecad6ef69e43c791"
ENV LOCATION="us"

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the application
CMD ["uv", "run", "streamlit", "run", "Home.py"]