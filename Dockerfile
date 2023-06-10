# Use an official Python runtime as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx

# Install OpenCV dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    libavformat-dev \
    libpq-dev
    
# Install CMake
RUN apt-get update && apt-get install -y cmake

# Copy the requirements file
COPY requirements.txt .

# Install the required Python packages
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Define the command to run the Flask application
CMD ["python", "app.py","0.0.0.0:5000"]
