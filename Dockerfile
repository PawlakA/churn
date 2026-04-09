# Use a slim Python image
FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy the rest of the source code
COPY src/ ./src/
COPY start.sh ./

# Make start script executable
RUN chmod +x start.sh

# Add src to Python path so imports like "from app.model.inference import predict" work
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8000 8501

# Run start script
CMD ["./start.sh"]