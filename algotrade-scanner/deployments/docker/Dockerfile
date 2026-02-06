# AlgoTrade Scanner - Production Dockerfile
# Multi-stage build for optimal image size

# Stage 1: Build TA-Lib
FROM python:3.11-slim as ta-lib-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Stage 2: Application
FROM python:3.11-slim

# Copy TA-Lib from builder
COPY --from=ta-lib-builder /usr/lib/libta_lib* /usr/lib/
COPY --from=ta-lib-builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata

# Expose Prometheus metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9090/metrics')" || exit 1

# Run the scanner
CMD ["python", "-m", "src.orchestrator.scheduler"]
