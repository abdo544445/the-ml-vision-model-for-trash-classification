#!/bin/bash
set -e

# Install dependencies
echo "Installing npm dependencies..."
npm install

# Install Python dependencies
echo "Installing Python dependencies..."
# Activate virtual environment if it exists, otherwise create it
if [ ! -d "venv" ]; then
    python -m venv venv
fi
. venv/bin/activate

pip install torch torchvision pillow numpy
pip install ultralytics  # YOLOv5

# Create lib directory if it doesn't exist
mkdir -p lib

# Download a sample image for testing
echo "Downloading sample image..."
curl -o lib/test-image.jpg https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg

# Build TypeScript code
echo "Building TypeScript code..."
npm run build

# Copy Python script to lib directory
echo "Copying Python script..."
cp src/ML_Processing_Module.py lib/

# Run the test
echo "Running test..."
node lib/test.js 