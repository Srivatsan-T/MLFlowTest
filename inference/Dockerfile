# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script into the container
COPY inference.py .

# Set environment variables (if any are required)
ENV MLFLOW_TRACKING_URI="http://ec2-52-27-180-229.us-west-2.compute.amazonaws.com:5000/"

# Run the inference script that pulls the model from MLflow and serves it
CMD ["python", "inference.py"]
