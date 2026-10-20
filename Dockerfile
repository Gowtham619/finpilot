FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FINPILOT_AUTOSEED=1 FINPILOT_SCHEDULER=1 FINPILOT_DB=sqlite:////tmp/finpilot.db
CMD ["sh", "-c", "uvicorn finpilot.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
