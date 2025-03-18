#!/usr/bin/env python3
"""
Specialized test script for plastic container detection
Particularly focused on detecting white plastic containers (jerry cans)
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
    from ML_Processing_Module import predict_image
except ImportError:
    print("Error: Could not import ML_Processing_Module.")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def detect_plastic_containers(image_path):
    """Direct plastic container detection without relying on YOLOv5"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        height, width = img_array.shape[0], img_array.shape[1]
        
        # Parameters for detection
        min_container_area = width * height * 0.05  # Minimum size to consider
        
        # Convert to HSV-like space for better color segmentation
        # We're looking for white/light-colored plastic containers
        
        # Simple brightness calculation
        brightness = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Threshold to find light-colored regions
        light_mask = brightness > 190  # Threshold for white/light colors
        
        # Find connected components in the binary mask
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(light_mask)
        
        # Extract component properties
        component_props = []
        for i in range(1, num_features + 1):
            component = (labeled_array == i)
            component_size = np.sum(component)
            
            if component_size > min_container_area:
                # Extract bounding box
                rows, cols = np.where(component)
                if len(rows) == 0 or len(cols) == 0:
                    continue
                    
                min_row, max_row = np.min(rows), np.max(rows)
                min_col, max_col = np.min(cols), np.max(cols)
                
                # Calculate aspect ratio and other properties
                component_height = max_row - min_row
                component_width = max_col - min_col
                aspect_ratio = component_width / component_height if component_height > 0 else 0
                
                # Filter components that match container properties
                # Typical plastic container/jerry can has reasonable aspect ratio
                if 0.4 < aspect_ratio < 2.5:
                    # Extract the candidate region
                    region = img_array[min_row:max_row, min_col:max_col]
                    
                    # Calculate region properties
                    region_color_std = np.std(region, axis=(0, 1))
                    
                    # Plastic containers tend to have uniform color
                    if region_color_std.mean() < 50:
                        # Get center point and region size
                        center_y = (min_row + max_row) // 2
                        center_x = (min_col + max_col) // 2
                        
                        component_props.append({
                            'bbox': [min_col, min_row, max_col, max_row],
                            'center': (center_x, center_y),
                            'size': component_size,
                            'aspect_ratio': aspect_ratio,
                            'color_std': region_color_std.mean()
                        })
        
        # Filter overlapping regions (keep the largest)
        filtered_components = []
        component_props.sort(key=lambda x: x['size'], reverse=True)
        
        for comp in component_props:
            x1, y1, x2, y2 = comp['bbox']
            overlap = False
            
            for existing in filtered_components:
                ex1, ey1, ex2, ey2 = existing['bbox']
                
                # Check for overlap
                if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                    # Calculate overlap area
                    overlap_width = min(x2, ex2) - max(x1, ex1)
                    overlap_height = min(y2, ey2) - max(y1, ey1)
                    overlap_area = overlap_width * overlap_height
                    
                    smaller_area = min(comp['size'], existing['size'])
                    
                    # If significant overlap, mark as overlapping
                    if overlap_area > smaller_area * 0.5:
                        overlap = True
                        break
            
            if not overlap:
                filtered_components.append(comp)
        
        # Convert to normalized coordinates and format like YOLOv5 output
        predictions = []
        for i, comp in enumerate(filtered_components):
            x1, y1, x2, y2 = comp['bbox']
            x = x1 / width
            y = y1 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Calculate approximate volume
            pixel_to_real_world_ratio = 0.01  # Placeholder
            real_width = w * width * pixel_to_real_world_ratio
            real_height = h * height * pixel_to_real_world_ratio
            estimated_depth = real_width * 0.7
            estimated_volume = real_width * real_height * estimated_depth
            
            predictions.append({
                'class': 'plastic',
                'detailed_class': 'plastic container',
                'confidence': min(0.95, 0.7 + (50 - comp['color_std']) / 100),  # Higher confidence for more uniform color
                'bbox': [x, y, w, h],
                'estimated_volume': round(estimated_volume, 2),
                'direct_detection': True  # Mark as directly detected
            })
            
        return predictions
    except Exception as e:
        print(f"Error in direct detection: {str(e)}")
        return []

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
        direct_marker = "*" if pred.get('direct_detection', False) else ""
        label = f"{pred['detailed_class']} ({pred['class']}): {pred['confidence']:.2f}{direct_marker}"
        
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

def main():
    parser = argparse.ArgumentParser(description='Test plastic container detection')
    parser.add_argument('--image', type=str, required=True, help='Path to the test image')
    parser.add_argument('--output', type=str, help='Path to save the annotated image (optional)')
    parser.add_argument('--direct-only', action='store_true', help='Use only direct detection, not YOLOv5')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
    
    print(f"Processing image: {args.image}")
    
    all_predictions = []
    
    # Run YOLOv5 detection first (unless --direct-only is specified)
    if not args.direct_only:
        try:
            start_time = time.time()
            yolo_predictions = predict_image(args.image)
            elapsed_time = time.time() - start_time
            
            print(f"YOLOv5 detection completed in {elapsed_time:.2f} seconds")
            print(f"Found {len(yolo_predictions)} objects using YOLOv5:")
            
            # Display predictions
            for i, pred in enumerate(yolo_predictions):
                print(f"  {i+1}. {pred['detailed_class']} (Type: {pred['class']}, Confidence: {pred['confidence']:.2f})")
            
            all_predictions.extend(yolo_predictions)
            
        except Exception as e:
            print(f"Error during YOLOv5 processing: {str(e)}")
    
    # Now run direct detection for plastic containers
    try:
        start_time = time.time()
        direct_predictions = detect_plastic_containers(args.image)
        elapsed_time = time.time() - start_time
        
        print(f"Direct detection completed in {elapsed_time:.2f} seconds")
        print(f"Found {len(direct_predictions)} plastic containers using direct detection:")
        
        # Display predictions
        for i, pred in enumerate(direct_predictions):
            print(f"  {i+1}. {pred['detailed_class']} (Type: {pred['class']}, Confidence: {pred['confidence']:.2f})")
        
        # Add direct detections that don't significantly overlap with YOLOv5 detections
        if not args.direct_only and len(yolo_predictions) > 0:
            for direct_pred in direct_predictions:
                dx, dy, dw, dh = direct_pred['bbox']
                
                # Check for overlap with existing YOLOv5 predictions
                overlaps = False
                for yolo_pred in yolo_predictions:
                    yx, yy, yw, yh = yolo_pred['bbox']
                    
                    # Calculate intersection
                    ix = max(dx, yx)
                    iy = max(dy, yy)
                    iw = min(dx + dw, yx + yw) - ix
                    ih = min(dy + dh, yy + yh) - iy
                    
                    if iw > 0 and ih > 0:
                        # Calculate overlap area
                        overlap_area = iw * ih
                        direct_area = dw * dh
                        yolo_area = yw * yh
                        
                        # If significant overlap, don't add this detection
                        if overlap_area > 0.3 * min(direct_area, yolo_area):
                            overlaps = True
                            break
                
                if not overlaps:
                    all_predictions.append(direct_pred)
        else:
            # If direct-only or no YOLOv5 predictions, add all direct detections
            all_predictions.extend(direct_predictions)
            
        # If running in direct-only mode, only use direct predictions
        if args.direct_only:
            all_predictions = direct_predictions
        
        # Draw predictions on the image
        print(f"Total predictions after combining methods: {len(all_predictions)}")
        output_path = args.output if args.output else None
        draw_predictions(args.image, all_predictions, output_path)
        
    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 