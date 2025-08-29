FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

# Update pip and install system dependencies
RUN apt-get update && \
    apt-get install -y libpq-dev build-essential && \
    python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

WORKDIR /app/src

CMD ["python", "main.py"]
