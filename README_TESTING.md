# Trash Detection Testing Guide

This guide will help you test the trash detection model and web interface we've implemented.

## Overview

Our system can detect various types of trash (plastic, paper, metal, glass, organic, mixed) from:
- Uploaded images
- Uploaded videos
- Webcam feed
- ESP32-CAM feed (if available)

It provides both a server-side and browser-based processing mode.

## Getting Started

### Prerequisites

- Python 3.7+ installed
- A virtual environment (venv)
- Web browser
- Webcam (optional, for webcam testing)
- ESP32-CAM (optional, for ESP32-CAM testing)

### Running the Test Server

We've created a test script that automates the setup and testing process:

```bash
# Make the test script executable (if not already)
chmod +x test_server.py

# Run the test script
./test_server.py
```

This script will:
1. Verify the virtual environment exists
2. Install required packages
3. Start the Flask server
4. Open your web browser to the test page

If you prefer to run the server manually:

```bash
# Activate the virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate  # On Windows

# Install required packages
pip install torch torchvision flask flask-cors pillow ultralytics

# Run the server
python server.py
```

Then open your browser to `http://localhost:8080`

## Testing the Website

### 1. Processing Modes

The website supports two processing modes:
- **Server Processing**: Uses the YOLOv5 model running on the server (more accurate for trash detection)
- **Browser Processing**: Uses TensorFlow.js running directly in the browser (no server needed)

Start by selecting your preferred processing mode at the top of the page.

### 2. Testing with Images

1. Click the "Image Upload" button
2. Click "Select Image" and choose an image with trash items
3. Click "Detect Objects"
4. Review the detection results
5. Optionally, click "Download Processed Image" to save the annotated image

### 3. Testing with Video

1. Click the "Video Upload" button
2. Click "Select Video" and choose a video with trash items
3. Click "Start Detection"
4. Review the detection results
5. Click "Stop Detection" when finished
6. Optionally, click "Download Processed Video" to save the video with annotations

### 4. Testing with Webcam

1. Click the "Use Webcam" button
2. Click "Start Webcam" and allow camera access
3. Click "Start Detection"
4. Move various trash items in front of the camera
5. Review the detection results in real-time
6. Click "Stop Detection" when finished

### 5. Testing with ESP32-CAM (if available)

1. Click the "ESP32-CAM Feed" button
2. Enter the IP address of your ESP32-CAM device
3. Click "Connect"
4. Once connected, click "Start Detection"
5. Review the detection results
6. Click "Stop Detection" when finished

## What to Expect

For each detected trash item, you should see:
- A colored bounding box around the item
- A label indicating the type of trash (plastic, paper, metal, etc.)
- The confidence level of the detection
- An estimated volume (approximate)

The results section will show:
- The processing mode being used
- Total estimated volume of all detected trash
- A breakdown of each trash type detected, with counts and volumes

## Troubleshooting

If you encounter issues:

1. **Server won't start**:
   - Check if port 8080 is already in use
   - Ensure you have all required dependencies installed
   - Check the log file: `trash_detection_server.log`

2. **No detections**:
   - Try using images with clearer trash items
   - Try using the server-side processing mode
   - Ensure good lighting if using webcam or ESP32-CAM

3. **Browser-based processing not working**:
   - Make sure you're using a modern browser (Chrome, Edge, Firefox)
   - Check the browser console for errors
   - Try disabling browser extensions that might interfere with TensorFlow.js

4. **Model loading errors**:
   - The system will fall back to a general-purpose model if the trash-specific model fails to load
   - Check internet connectivity, as the model is downloaded from Hugging Face

## Extending the Testing

Feel free to test with different types of trash items to see how well the system performs. The model should be able to identify various common trash types, but it may have limitations with certain unusual items or in poor lighting conditions.

You can also test the volume estimation by using items of known volume and comparing them to the system's estimates. Note that the volume estimation is approximate and based on 2D images, so there will be some inaccuracy.

## Feedback and Improvements

As you test, note any issues or potential improvements. The system can be further improved by:
- Fine-tuning the model on more specific trash datasets
- Improving the volume estimation algorithms
- Adding calibration features for more accurate measurements
- Expanding the types of trash that can be detected 