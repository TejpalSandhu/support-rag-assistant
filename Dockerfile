# Backend: the FastAPI app around rag_assistant.py

FROM python:3.12-slim

# no .pyc files in the image, and print logs straight away
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

WORKDIR /app

# dependencies first: this layer is reused until requirements.txt changes,
# so code-only changes rebuild in seconds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_assistant.py api.py ./

# run as a normal user, not root, and give it the data folder
RUN useradd --create-home app && mkdir -p /data && chown app:app /data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# one worker: upload status is kept in this process's memory
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
