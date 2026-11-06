FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SEED_DATA_DIR=/data/seed
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/seed-browsers
WORKDIR /app
COPY RAG/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt && useradd --uid 10001 --create-home seed
RUN python -m playwright install --with-deps --only-shell chromium && chmod -R a+rX /opt/seed-browsers
COPY RAG /app/RAG
COPY Interface/dist /app/Interface/dist
RUN mkdir -p /data/seed && chown -R seed:seed /data/seed
USER seed
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready')"
CMD ["python", "-m", "RAG", "serve", "--host", "0.0.0.0", "--port", "8000"]
