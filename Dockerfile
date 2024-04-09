# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenSlide and basic build tools
RUN apt-get update && \
    apt-get install -y build-essential openslide-tools libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the local directory contents into the container at /app
COPY ./ /app/

# Install Python dependencies
# Make sure you have a requirements.txt file in your directory with all the necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and switch to it for security reasons
RUN adduser --disabled-password --gecos '' myuser
USER myuser

# No need to expose a port unless your application provides a web service
EXPOSE 8888

ENTRYPOINT ["python", "main.py"]
