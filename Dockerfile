# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Verify uv installation
RUN uv --version

# Copy project files
COPY . .

# Install dependencies using uv
RUN pip install -r requirements.txt


EXPOSE 5555


# Run the Gunicorn server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5555", "app:app"]