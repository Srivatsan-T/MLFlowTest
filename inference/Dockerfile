# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script into the container
COPY inference.py .

# Copy the folder to the target directory (e.g., /app/myfolder)
COPY ../modules /app/modules

# Set environment variables (if any are required)
ENV MLFLOW_TRACKING_URI="http://localhost:5000/"

# Run the inference script that pulls the model from MLflow and serves it
CMD ["python", "inference.py"]
