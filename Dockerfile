# Use Python 3.10 as base image
FROM python:3.10

# Set working directory in container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "mock_api:app", "--host", "0.0.0.0", "--port", "8000"]