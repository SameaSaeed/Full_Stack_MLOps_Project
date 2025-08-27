# Use a slim Python base image for efficiency
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code from `src/` to `/app`
COPY src/ .

# Copy the model file into the correct location
COPY models/trained_model.pkl /app/models/trained_model.pkl

# Launch with Uvicorn on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
