# Using light weight python 
FROM python:3.10-slim

# Set the working directory
WORKDIR /app 

# Copy the requirements and install dependencies
COPY requirements.txt . 
RUN pip install -r requirements.txt 

# Copy entire application
COPY . .

# Expose Port for flask 
EXPOSE 5000 

# Run the Flask API 
CMD ["python", "-m", "api.app"] 

# use:
#   1. docker build -t sentiment-pipeline -f docker/Dockerfile .
#   2. docker run -p 5000:5000 -v ${PWD}\data:/app/data sentiment-pipeline