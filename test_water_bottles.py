#!/usr/bin/env python3
"""
Test script for water bottle and plastic bottle detection accuracy
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Add the functions directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(current_dir, 'functions', 'src')
sys.path.insert(0, functions_dir)

try:
    from ML_Processing_Module import predict_image, WaterBottleDetector
except ImportError:
    print("Error: Could not import ML_Processing_Module.")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def draw_predictions(image_path, predictions, output_path=None):
    """Draw bounding boxes and labels on the image"""
    # Load the image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Define colors for different waste types
    colors = {
        'plastic': (255, 0, 0),      # Red
        'glass': (0, 255, 0),        # Green
        'metal': (0, 0, 255),        # Blue
        'paper': (255, 255, 0),      # Yellow
        'organic': (0, 255, 255),    # Cyan
        'electronic': (255, 0, 255), # Magenta
        'mixed': (128, 128, 128)     # Gray
    }
    
    width, height = img.size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each prediction
    for pred in predictions:
        # Extract bounding box
        x, y, w, h = pred['bbox']
        x1, y1 = int(x * width), int(y * height)
        x2, y2 = int((x + w) * width), int((y + h) * height)
        
        # Get color based on waste type
        color = colors.get(pred['class'], (128, 128, 128))
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label text
        label = f"{pred['detailed_class']} ({pred['confidence']:.2f})"
        
        # Draw label background
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width, y1], fill=color)
        
        # Draw label text
        draw.text((x1, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    # Save or display the image
    if output_path:
        img.save(output_path)
        print(f"Annotated image saved to {output_path}")
    else:
        img.show()

def test_water_bottle_direct(image_path):
    """Test the specialized water bottle detector directly"""
    img = Image.open(image_path).convert('RGB')
    
    # Try different crop sizes from the image
    width, height = img.size
    crops_to_test = [
        {"name": "Full image", "crop": (0, 0, width, height)},
        {"name": "Center crop", "crop": (width//4, height//4, 3*width//4, 3*height//4)},
        {"name": "Top half", "crop": (0, 0, width, height//2)},
        {"name": "Bottom half", "crop": (0, height//2, width, height)}
    ]
    
    results = []
    for crop_test in crops_to_test:
        crop_name = crop_test["name"]
        crop_box = crop_test["crop"]
        img_crop = img.crop(crop_box)
        
        # Test the water bottle detector
        is_bottle, confidence = WaterBottleDetector.is_water_bottle(img_crop, "bottle")
        
        results.append({
            "crop_name": crop_name, 
            "is_bottle": is_bottle, 
            "confidence": confidence
        })
        
        print(f"Crop: {crop_name}, Is Water Bottle: {is_bottle}, Confidence: {confidence:.2f}")
    
    return results

def test_material_analysis(image_path):
    """Perform full detection and analyze results for water bottles and plastic materials"""
    try:
        start_time = time.time()
        predictions = predict_image(image_path)
        elapsed_time = time.time() - start_time
        
        print(f"\nMaterial analysis completed in {elapsed_time:.2f} seconds")
        print(f"Found {len(predictions)} objects:")
        
        # Count materials
        material_counts = {}
        for pred in predictions:
            material = pred['class']
            material_counts[material] = material_counts.get(material, 0) + 1
            
            # Print detailed information
            print(f"  • {pred['detailed_class']} (Type: {pred['class']}, Confidence: {pred['confidence']:.2f})")
        
        # Print material statistics
        print("\nMaterial Statistics:")
        for material, count in material_counts.items():
            print(f"  • {material.capitalize()}: {count} object(s)")
            
        return predictions
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Test water bottle and plastic detection')
    parser.add_argument('--image', type=str, required=True, help='Path to the test image')
    parser.add_argument('--output', type=str, help='Path to save the annotated image (optional)')
    parser.add_argument('--direct-test', action='store_true', help='Run direct water bottle detector test')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
    
    print(f"Processing image: {args.image}")
    
    # Run direct water bottle detector test if requested
    if args.direct_test:
        print("\n=== WATER BOTTLE DIRECT TEST ===")
        test_water_bottle_direct(args.image)
    
    # Run full material analysis
    print("\n=== FULL MATERIAL ANALYSIS ===")
    predictions = test_material_analysis(args.image)
    
    # Draw predictions on the image
    if predictions:
        output_path = args.output if args.output else None
        draw_predictions(args.image, predictions, output_path)

if __name__ == "__main__":
    main() 