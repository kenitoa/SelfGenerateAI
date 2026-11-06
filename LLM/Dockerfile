FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY RAG/requirements.txt /tmp/rag-requirements.txt
COPY LLM/requirements.txt /tmp/llm-requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.9,<3' \
 && pip install --no-cache-dir -r /tmp/rag-requirements.txt -r /tmp/llm-requirements.txt
COPY RAG /app/RAG
COPY LLM /app/LLM
COPY Interface/dist /app/Interface/dist
ENV SEED_LLM_DATA_DIR=/data/llm SEED_DATA_DIR=/data/rag
EXPOSE 8000
CMD ["python","-m","LLM","serve","--integrated","--host","0.0.0.0","--port","8000","--rag-data-dir","/data/rag"]
