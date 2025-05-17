# Use Python 3.10 as base image
     FROM python:3.10

     # Set working directory in container
     WORKDIR /app

     # Copy requirements and install dependencies
     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt

     # Copy entire application
     COPY . .

     # Expose Railway's dynamic port
     EXPOSE 8080

     # Run FastAPI with Uvicorn using PORT env variable
     CMD ["sh", "-c", "uvicorn mock_api:app --host 0.0.0.0 --port $PORT"]