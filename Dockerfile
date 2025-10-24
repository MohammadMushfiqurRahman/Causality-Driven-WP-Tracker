# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some python packages (e.g., shap)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./src /app/src
COPY ./models /app/models

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
# This will be updated once the FastAPI app is created
CMD ["echo", "FastAPI app not yet configured. Run 'uvicorn api.main:app --host 0.0.0.0 --port 8000'"]