FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY aml ./aml
EXPOSE 8000 8501
CMD ["uvicorn", "aml.api:app", "--host", "0.0.0.0", "--port", "8000"]
