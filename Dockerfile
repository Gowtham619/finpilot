FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
ARG WITH_OCR=1   # 0 skips OCR (about 200 MB smaller); CSV and text PDFs still work
COPY requirements.txt requirements-ocr.txt ./
RUN pip install -r requirements.txt && if [ "$WITH_OCR" = "1" ]; then pip install -r requirements-ocr.txt; fi

# eval/ is required at runtime (the app imports eval.generate for the demo personas)
COPY . .

# Non-root user; /data is the writable home of the SQLite file
RUN useradd --create-home --uid 10001 finpilot \
    && mkdir -p /data && chown -R finpilot:finpilot /data /app
USER finpilot

ENV FINPILOT_AUTOSEED=1 \
    FINPILOT_SCHEDULER=1 \
    FINPILOT_DB=sqlite:////data/finpilot.db \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PORT','8000'), timeout=4).read()"]

CMD ["sh", "-c", "exec uvicorn finpilot.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
