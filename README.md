# AI Trash Detection System

An advanced AI-powered system for detecting and classifying different types of trash and recyclable materials. The system uses computer vision and machine learning to identify various objects and their materials, helping with waste management and recycling efforts.

## Features

- Real-time object detection and material classification
- Support for multiple input sources:
  - Image upload
  - Video upload
  - Webcam feed
  - ESP32-CAM integration
- Material classification for:
  - Plastic
  - Glass
  - Metal
  - Paper
  - Organic waste
  - Electronics
- Volume estimation for detected objects
- Shape analysis for improved accuracy
- Web interface with real-time visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdo544445/trash-detection-ai.git
cd trash-detection-ai
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python server.py
```

4. Open your web browser and navigate to:
```
http://localhost:8080
```

## Usage

1. Choose your input source (image, video, webcam, or ESP32-CAM)
2. Upload or capture the content you want to analyze
3. The system will automatically detect objects and classify their materials
4. Results will be displayed with bounding boxes and material classifications
5. Download the processed results if needed

## Technical Details

- Built with Python and Flask
- Uses YOLOv5 for object detection
- Implements advanced material classification algorithms
- Real-time processing capabilities
- Cross-platform compatibility

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 