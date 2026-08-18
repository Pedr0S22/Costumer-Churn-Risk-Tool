FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/churn-model_v1.joblib

EXPOSE 8000

# By default, run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
