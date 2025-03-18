from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import logging
import base64
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trash_detection_server.log')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting Enhanced Trash Detection Server v2.0")

# Add the functions directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(current_dir, 'functions', 'src')
sys.path.insert(0, functions_dir)

try:
    logger.info("Attempting to import Enhanced ML_Processing_Module...")
    start_time = time.time()
    from ML_Processing_Module import predict_image, MaterialAnalyzer, WaterBottleDetector
    elapsed_time = time.time() - start_time
    logger.info(f"Successfully imported ML_Processing_Module. Model loading took {elapsed_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error importing ML_Processing_Module: {str(e)}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Detailed error information: {e.__class__.__name__}")
    for line in str(e).split('\n'):
        logger.error(f"> {line}")
    # Continue without raising to try to gracefully handle this situation
    logger.warning("Continuing without model, detection capabilities will be limited")

from werkzeug.utils import secure_filename
import tempfile
from PIL import Image
import io
import json

app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = tempfile.gettempdir()
# Expanded list of allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'avif', 'tiff', 'tif', 'heic', 'heif'}

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def convert_to_compatible_format(input_path):
    """Convert problematic image formats to standard JPG for processing"""
    try:
        # Try to open the image with PIL
        with Image.open(input_path) as img:
            # Convert to RGB (in case of RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as temporary JPG file
            temp_jpg_path = input_path + '.jpg'
            img.save(temp_jpg_path, 'JPEG', quality=95)
            
            # Remove original file
            if os.path.exists(input_path):
                os.remove(input_path)
                
            return temp_jpg_path
    except Exception as e:
        logger.error(f"Error converting image: {str(e)}")
        return input_path  # Return original path if conversion fails

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect_objects():
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info("Received detection request")
    
    if 'image' not in request.files:
        logger.error("No image file in request")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # For video frames, we'll use a default filename if none is provided
    filename = file.filename if file.filename else 'frame.jpg'
    logger.info(f"Processing file: {filename}")
    
    # Try to process even if the extension isn't in our list
    # We'll convert it later if needed
    filename = secure_filename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Convert image to a compatible format if needed
        if not allowed_file(filename):
            logger.warning(f"Non-standard image format: {filename}, attempting conversion")
            filepath = convert_to_compatible_format(filepath)
            
        logger.info(f"Processing image for trash detection: {filepath}")
        
        # Process the image with enhanced detection
        start_time = time.time()
        predictions = predict_image(filepath)
        elapsed_time = time.time() - start_time
        logger.info(f"Enhanced detection completed in {elapsed_time:.2f} seconds. Found {len(predictions)} items.")
        
        # Ensure all predictions have required fields for frontend compatibility
        shapes_detected = []
        for pred in predictions:
            # Collect information about detected shapes
            if 'shape_info' in pred:
                shape_type = pred['shape_info']
                shapes_detected.append(shape_type)
                logger.info(f"Detected shape: {shape_type} for {pred.get('detailed_class', 'unknown object')}")
            
            # Ensure all predictions have basic required fields
            if 'class' not in pred:
                if 'detailed_class' in pred:
                    # Try to extract material type from detailed class
                    parts = pred['detailed_class'].split()
                    if parts and parts[0] in ['plastic', 'glass', 'metal', 'paper', 'organic', 'electronic']:
                        pred['class'] = parts[0]
                    else:
                        pred['class'] = 'mixed'
                else:
                    pred['class'] = 'mixed'
                    
            # Ensure detailed_class exists
            if 'detailed_class' not in pred:
                pred['detailed_class'] = pred.get('class', 'unknown')
                
            # Ensure confidence is a number between 0-1
            if 'confidence' not in pred or not isinstance(pred['confidence'], (int, float)):
                pred['confidence'] = 0.5
            elif isinstance(pred['confidence'], str):
                try:
                    pred['confidence'] = float(pred['confidence'])
                except:
                    pred['confidence'] = 0.5
                    
            # Ensure bounding box is properly formatted
            if 'bbox' not in pred or not isinstance(pred['bbox'], list) or len(pred['bbox']) != 4:
                pred['bbox'] = [0.1, 0.1, 0.2, 0.2]  # Default small bounding box
                
            # Ensure estimated volume exists
            if 'estimated_volume' not in pred:
                # Calculate a simple volume based on bounding box
                [x, y, w, h] = pred['bbox']
                pred['estimated_volume'] = round(w * h * 0.7, 2)
                
            # Format confidence value for logging
            confidence_str = f"{pred['confidence']:.2f}" if isinstance(pred['confidence'], float) else pred['confidence']
            logger.info(f"Detected: {pred['detailed_class']} as {pred['class']} with confidence {confidence_str}")
        
        # Add extra metadata for the client
        response_data = {
            'predictions': predictions,
            'metadata': {
                'processing_time': elapsed_time,
                'model_version': 'Enhanced Material Classifier v2.1',
                'materials_detected': list(set(pred['class'] for pred in predictions)),
                'shapes_detected': shapes_detected if shapes_detected else ["standard"]
            }
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def api_detect_objects():
    """Alternative API endpoint for compatibility with older clients"""
    return detect_objects()

@app.route('/api/analyze_bottle', methods=['POST'])
def analyze_bottle():
    """Special endpoint for analyzing if an image contains a plastic bottle"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename if file.filename else 'bottle.jpg')
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Convert image if needed
        if not allowed_file(filename):
            filepath = convert_to_compatible_format(filepath)
            
        # Open the image and analyze it
        img = Image.open(filepath).convert('RGB')
        is_bottle, confidence = WaterBottleDetector.is_water_bottle(img, "bottle")
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'is_plastic_bottle': is_bottle,
            'confidence': confidence,
            'recommendation': 'This should be recycled as plastic' if is_bottle else 'Unable to confirm if this is a plastic bottle'
        })
    except Exception as e:
        logger.error(f"Error analyzing bottle: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'message': 'Enhanced trash detection server is running',
        'version': 'v2.0'
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == '__main__':
    # Print current directory and Python path for debugging
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Static folder: {app.static_folder}")
    
    # Listen on all interfaces so it's accessible from other devices
    try:
        app.run(debug=True, host='0.0.0.0', port=8080)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1) 