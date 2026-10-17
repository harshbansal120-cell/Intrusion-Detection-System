# Hugging Face Spaces (Docker SDK) — this is what your friend needs to
# put in the SAME folder as app.py when creating the Space.
# HF Spaces expects the app to listen on port 7860 by default.

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
