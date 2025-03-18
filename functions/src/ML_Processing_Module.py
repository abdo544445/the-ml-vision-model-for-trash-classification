import torch
import torchvision
from PIL import Image
import json
import sys
import os
import numpy as np
from collections import Counter
import math

# ====================================================
# OBJECT DETECTION MODEL LOADING
# ====================================================

def load_model():
    """Load the best available object detection model"""
    
    models_to_try = [
        # Option 1: Load the standard YOLOv5 model which is known to be reliable
        {"type": "standard", "function": lambda: torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)},
        
        # Option 2: Try local YOLOv5 model
        {"type": "local", "path": 'yolov5s.pt', 
         "function": lambda path=None: torch.hub.load('ultralytics/yolov5', 'custom', path=path)},
        
        # Option 3: Try TACO-specific model if available
        {"type": "taco", "path": 'taco_yolov5s.pt', 
         "function": lambda path=None: torch.hub.load('ultralytics/yolov5', 'custom', path=path)}
    ]
    
    for model_config in models_to_try:
        try:
            if model_config["type"] == "standard":
                model = model_config["function"]()
                print(f"Successfully loaded standard YOLOv5s model")
                return model
            else:
                path = model_config["path"]
                if os.path.exists(path):
                    model = model_config["function"](path)
                    print(f"Loaded {model_config['type']} model from {path}")
                    return model
        except Exception as e:
            print(f"Error loading {model_config['type']} model: {str(e)}")
            continue
    
    # If we've tried all models and none worked, raise an exception
    raise Exception("Failed to load any object detection model")

# Load the model
try:
    MODEL = load_model()
    MODEL.eval()
except Exception as e:
    print(f"Critical error loading model: {str(e)}")

# ====================================================
# MATERIAL CLASSIFICATION SYSTEM 
# ====================================================

class MaterialAnalyzer:
    """Advanced material analysis system that combines multiple techniques to identify materials"""
    
    # ---- REFERENCE DATA ----
    
    # Material-specific color profiles in RGB
    REFERENCE_COLORS = {
        "plastic": [
            [255, 255, 255],  # White plastic (very common for containers)
            [245, 245, 245],  # Off-white plastic
            [200, 200, 200],  # Light gray plastic
            [0, 0, 255],      # Blue plastic
            [255, 255, 0],    # Yellow plastic
            [0, 255, 0],      # Green plastic
            [255, 0, 0],      # Red plastic
            [0, 0, 0],        # Black plastic
            [255, 165, 0],    # Orange plastic
            [0, 128, 255],    # Light blue (water bottles)
            [192, 192, 192],  # Silver gray (some plastic packaging)
        ],
        "glass": [
            [200, 230, 255],  # Clear/blue-tinted glass
            [220, 220, 220],  # Clear glass
            [150, 200, 150],  # Green glass
            [200, 150, 50],   # Amber/brown glass
            [100, 160, 180],  # Blue glass
        ],
        "metal": [
            [192, 192, 192],  # Silver/aluminum
            [128, 128, 128],  # Gray metal
            [212, 175, 55],   # Gold/brass colored
            [184, 115, 51],   # Copper/bronze
        ],
        "paper": [
            [255, 250, 240],  # Off-white paper
            [245, 222, 179],  # Beige/cardboard
            [255, 255, 240],  # Ivory
            [220, 220, 220],  # Light gray paper/cardboard
            [200, 180, 150],  # Brown cardboard
        ]
    }
    
    # Known objects and their typical materials
    OBJECT_MATERIALS = {
        # Plastic objects
        "water bottle": "plastic",
        "soda bottle": "plastic",
        "plastic bottle": "plastic",
        "plastic bag": "plastic",
        "straw": "plastic",
        "cup": "plastic",
        "plastic cup": "plastic",
        "food container": "plastic",
        "yogurt": "plastic",
        "detergent": "plastic",
        "shampoo": "plastic",
        "bucket": "plastic",
        "toys": "plastic",
        "pen": "plastic",
        "toothbrush": "plastic",
        "milk jug": "plastic",
        "soap dispenser": "plastic",
        "tupperware": "plastic",
        "takeout container": "plastic",
        "jerry can": "plastic",
        "jug": "plastic",
        
        # Glass objects
        "wine bottle": "glass",
        "beer bottle": "glass",
        "glass bottle": "glass",
        "wine glass": "glass",
        "drinking glass": "glass",
        "jar": "glass",
        "glass jar": "glass",
        "perfume": "glass",
        "mirror": "glass",
        "window": "glass",
        
        # Metal objects
        "soda can": "metal",
        "beer can": "metal",
        "can": "metal",
        "tin can": "metal",
        "aluminum can": "metal",
        "fork": "metal",
        "knife": "metal",
        "spoon": "metal",
        "foil": "metal",
        "metal lid": "metal",
        "coin": "metal",
        "key": "metal",
        "paperclip": "metal",
        "screws": "metal",
        "nails": "metal",
        "wire": "metal",
        
        # Paper objects
        "newspaper": "paper",
        "magazine": "paper",
        "book": "paper",
        "cardboard": "paper",
        "cardboard box": "paper",
        "paper bag": "paper",
        "tissue": "paper",
        "napkin": "paper",
        "paper towel": "paper",
        "receipt": "paper",
        "envelope": "paper",
        "notebook": "paper",
        "carton": "paper",
        "milk carton": "paper",
        "juice carton": "paper",
        "egg carton": "paper",
        
        # Organic objects
        "fruit": "organic",
        "vegetable": "organic",
        "food": "organic",
        "banana": "organic",
        "apple": "organic",
        "orange": "organic",
        "leaves": "organic",
        "branch": "organic",
        "grass": "organic",
        "coffee grounds": "organic",
        "tea bag": "organic",
        "plant": "organic",
        "flower": "organic",
        
        # Electronic objects
        "phone": "electronic",
        "cell phone": "electronic",
        "smartphone": "electronic",
        "laptop": "electronic",
        "computer": "electronic",
        "tablet": "electronic",
        "tv": "electronic",
        "remote": "electronic",
        "battery": "electronic",
        "charger": "electronic",
        "cable": "electronic",
        "earbuds": "electronic",
        "headphones": "electronic",
    }
    
    # Material property patterns (typical visual characteristics of materials)
    MATERIAL_PROPERTIES = {
        "plastic": {
            "surface_smoothness": "high",      # Plastic usually has smooth surfaces
            "color_uniformity": "high",        # Plastic often has uniform color
            "reflectivity": "low_to_medium",   # Some plastic is matte, some is glossy
            "transparency": "variable",        # Can be transparent, translucent, or opaque
            "texture_pattern": "none",         # Usually no visible texture pattern
            "edge_sharpness": "medium",        # Plastic edges can be defined but not too sharp
        },
        "glass": {
            "surface_smoothness": "very_high",  # Glass is very smooth
            "color_uniformity": "medium",       # Glass can have slight color variations
            "reflectivity": "high",             # Glass is highly reflective
            "transparency": "high",             # Glass is often transparent or translucent
            "texture_pattern": "none",          # No texture pattern
            "edge_sharpness": "high",           # Glass edges are usually very defined
        },
        "metal": {
            "surface_smoothness": "high",       # Metal surfaces are smooth
            "color_uniformity": "medium",       # Some variation due to light reflection
            "reflectivity": "very_high",        # Metals are highly reflective
            "transparency": "none",             # Metals are not transparent
            "texture_pattern": "variable",      # Some metals have patterns (brushed aluminum)
            "edge_sharpness": "very_high",      # Metal edges are typically sharp
        },
        "paper": {
            "surface_smoothness": "low",        # Paper has textured surfaces
            "color_uniformity": "medium",       # Paper has some color variations
            "reflectivity": "very_low",         # Paper is usually not reflective
            "transparency": "none",             # Paper isn't transparent (except thin types)
            "texture_pattern": "visible",       # Paper often has visible fiber patterns
            "edge_sharpness": "low",            # Paper edges aren't sharp
        }
    }
    
    # ---- FEATURE EXTRACTION ----
    
    @staticmethod
    def extract_features(img_crop):
        """Extract comprehensive set of visual features from an image crop"""
        
        # Convert PIL image to numpy array
        img_array = np.array(img_crop)
        if len(img_array.shape) < 3:  # Handle grayscale images
            img_array = np.stack((img_array,) * 3, axis=-1)
        
        # Image dimensions
        height, width = img_array.shape[0], img_array.shape[1]
        if height == 0 or width == 0:
            # Return default features for empty image
            return {
                'avg_color': np.array([0, 0, 0]),
                'color_std': np.array([0, 0, 0]),
                'brightness_mean': 0,
                'brightness_std': 0,
                'edge_density': 0,
                'color_hist': np.zeros(24),
                'unique_colors_ratio': 0,
                'highlights_ratio': 0,
                'transparency_est': 0,
                'aspect_ratio': 1.0,
                'top_color_ratio': 0,
                'texture_pattern': 0,
                'entropy': 0
            }
            
        # Basic color statistics
        avg_color = np.mean(img_array, axis=(0, 1))
        color_std = np.std(img_array, axis=(0, 1))
        
        # Brightness features
        brightness = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2])
        brightness_mean = np.mean(brightness)
        brightness_std = np.std(brightness)
        
        # Edge detection (measure of texture)
        edges_x = np.abs(np.diff(brightness, axis=1, prepend=brightness[:,[0]]))
        edges_y = np.abs(np.diff(brightness, axis=0, prepend=brightness[[0],:]))
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        edge_density = np.mean(edge_magnitude) / 255.0
        
        # Color histogram (8 bins per channel = 24 total features)
        r_hist = np.histogram(img_array[:,:,0], bins=8, range=(0, 256))[0] / (width * height)
        g_hist = np.histogram(img_array[:,:,1], bins=8, range=(0, 256))[0] / (width * height)
        b_hist = np.histogram(img_array[:,:,2], bins=8, range=(0, 256))[0] / (width * height)
        color_hist = np.concatenate([r_hist, g_hist, b_hist])
        
        # Color diversity analysis
        pixels = img_array.reshape(-1, 3)
        pixel_count = len(pixels)
        
        # Use a more efficient method for large images
        if pixel_count > 100000:
            # Downsample by taking every nth pixel
            sample_rate = int(pixel_count / 100000) + 1
            pixels = pixels[::sample_rate]
            pixel_count = len(pixels)
        
        # Count unique colors with rounding to reduce noise
        rounded_pixels = (pixels / 8).astype(int) * 8  # Round to nearest 8
        unique_colors, counts = np.unique(rounded_pixels, axis=0, return_counts=True)
        unique_colors_ratio = len(unique_colors) / pixel_count
        
        # Highlights detection (reflections or bright spots)
        highlights_ratio = np.sum(brightness > 230) / (width * height)
        
        # Transparency estimation
        transparency_est = brightness_std * unique_colors_ratio
        
        # Shape features
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Color uniformity (how much of the image is the dominant color)
        if len(counts) > 0:
            top_color_ratio = counts[0] / np.sum(counts) if counts[0] > 0 else 0
        else:
            top_color_ratio = 0
            
        # Texture pattern detection
        # Calculate auto-correlation as a measure of repeating patterns
        texture_pattern = 0
        try:
            if width > 10 and height > 10:
                gray = brightness.astype(np.float32) / 255
                dx = np.diff(gray, axis=1)
                dy = np.diff(gray, axis=0)
                gx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
                gy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
                texture_pattern = np.mean(np.abs(gx) + np.abs(gy))
        except Exception:
            pass
            
        # Calculate image entropy (measure of information/complexity)
        entropy = 0
        try:
            hist = np.histogram(brightness.flatten(), bins=256, range=(0, 256))[0]
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
        except Exception:
            pass
        
        # Return all features as a dictionary
        return {
            'avg_color': avg_color,
            'color_std': color_std,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'edge_density': edge_density,
            'color_hist': color_hist,
            'unique_colors_ratio': unique_colors_ratio,
            'highlights_ratio': highlights_ratio,
            'transparency_est': transparency_est,
            'aspect_ratio': aspect_ratio,
            'top_color_ratio': top_color_ratio,
            'texture_pattern': texture_pattern,
            'entropy': entropy
        }

    # ---- MATERIAL CLASSIFICATION METHODS ----
    
    @staticmethod
    def color_similarity(color, reference_colors):
        """Calculate similarity to reference material colors"""
        color = np.array(color)
        min_dist = float('inf')
        
        for ref_color in reference_colors:
            # Use weighted Euclidean distance in RGB space
            dist = np.sqrt(((color - ref_color)**2).sum())
            min_dist = min(min_dist, dist)
            
        # Convert distance to similarity score (0-1)
        similarity = max(0, 1 - (min_dist / 441.7))  # 441.7 = max possible distance in RGB space
        return similarity
    
    @staticmethod
    def classify_by_material_name(class_name):
        """Check if the class name directly indicates the material"""
        class_name = class_name.lower()
        
        # Check if material name is directly in the class name
        materials = ["plastic", "glass", "metal", "paper", "cardboard", "wooden", "organic", "electronic"]
        for material in materials:
            if material in class_name:
                if material == "cardboard":
                    return "paper"
                if material == "wooden":
                    return "organic"
                return material
                
        # Check against common objects with known materials
        for obj, material in MaterialAnalyzer.OBJECT_MATERIALS.items():
            if obj in class_name or class_name in obj:
                return material
                
        return None
    
    @staticmethod
    def classify_material(img_crop, class_name, shape_ratio=None):
        """
        Main material classification method using multiple sources of evidence
        
        Parameters:
            img_crop: PIL Image cropped to the object
            class_name: Object class name from the detector
            shape_ratio: Optional width-to-height ratio
            
        Returns:
            material_type: The identified material type
            confidence: Confidence score (0-1) for the classification
        """
        class_name = class_name.lower()
        
        # Try to classify by name first
        material_by_name = MaterialAnalyzer.classify_by_material_name(class_name)
        if material_by_name:
            # If it's a direct material name match, high confidence
            if material_by_name in ["plastic", "glass", "metal", "paper"]:
                return material_by_name, 0.95
                
        # Extract visual features
        features = MaterialAnalyzer.extract_features(img_crop)
        
        # Initialize material scores dictionary with all possible materials
        scores = {
            'plastic': 0.0,
            'glass': 0.0,
            'metal': 0.0,
            'paper': 0.0,
            'organic': 0.0,
            'electronic': 0.0,
            'mixed': 0.0
        }
        
        # If we already have a material from the name, give it a starting score
        if material_by_name:
            scores[material_by_name] += 4.0

        # ---- EVALUATE COLOR SIMILARITY ----
        for material, ref_colors in MaterialAnalyzer.REFERENCE_COLORS.items():
            similarity = MaterialAnalyzer.color_similarity(features['avg_color'], ref_colors)
            scores[material] += similarity * 2.0  # Weight color similarity
        
        # ---- EVALUATE SPECIAL CASES ----
        
        # Special case 1: Water bottles and transparent plastic bottles
        if "bottle" in class_name:
            # Check for water bottle characteristics
            if (features['brightness_mean'] > 180 and 
                features['transparency_est'] > 5 and 
                features['top_color_ratio'] < 0.6):
                
                # Probable water bottle - clean, transparent
                scores['plastic'] += 3.0
                
            # Check for colored plastic bottles (soda, etc.)
            elif features['color_std'].mean() < 40 and features['highlights_ratio'] < 0.1:
                scores['plastic'] += 2.0
                
        # Special case 2: White plastic containers (like those in the image)
        if "container" in class_name or shape_ratio > 0.5:
            if (features['brightness_mean'] > 170 and
                features['color_std'].mean() < 30 and
                features['highlights_ratio'] < 0.06):
                
                # Likely a white/light colored plastic container
                scores['plastic'] += 5.0

        # ---- EVALUATE MATERIAL PROPERTIES ----
        
        # Plastic indicators
        if (features['edge_density'] < 0.1 and 
            features['top_color_ratio'] > 0.6 and
            features['texture_pattern'] < 0.05):
            # Smooth surface, uniform color, low texture - typical of plastic
            scores['plastic'] += 2.0
        
        # Glass indicators
        if (features['transparency_est'] > 20 and 
            features['highlights_ratio'] > 0.08 and
            features['brightness_std'] > 40):
            # Transparent, reflective (highlights), variable brightness - glass characteristics
            scores['glass'] += 2.0
            
        # Metal indicators
        if (features['highlights_ratio'] > 0.1 and 
            features['edge_density'] > 0.05 and
            features['entropy'] > 6.0):
            # Reflective, defined edges, complex appearance - metal characteristics
            scores['metal'] += 2.0
            
        # Paper indicators
        if (features['edge_density'] > 0.08 and 
            features['highlights_ratio'] < 0.05 and
            features['texture_pattern'] > 0.03):
            # Textured, matte appearance - paper characteristics
            scores['paper'] += 2.0
            
        # Organic indicators
        if "food" in class_name or "fruit" in class_name or "vegetable" in class_name:
            scores['organic'] += 3.0
            
        # Electronic indicators
        if "phone" in class_name or "computer" in class_name or "electronic" in class_name:
            scores['electronic'] += 3.0
           
        # Get the highest scoring material
        best_material = max(scores.items(), key=lambda x: x[1])
        material_type = best_material[0]
        score = best_material[1]
        
        # Normalize score to a confidence value (0-1)
        confidence = min(0.98, score / 10)  # Cap at 0.98
        
        # Safety check - if confidence is too low, default to a reasonable guess based on class
        if confidence < 0.4:
            # Common objects that might be missed
            if "bottle" in class_name:
                material_type = "plastic"  # Most bottles are plastic
                confidence = 0.75
            elif class_name in ["cup", "container", "box"]:
                material_type = "plastic"  # Assume plastic by default
                confidence = 0.6
                
        return material_type, confidence

# ====================================================
# OBJECT TO MATERIAL CATEGORY MAPPING
# ====================================================

# This is a backup for when the material analyzer is uncertain
OBJECT_TO_WASTE_MAP = {
    # Plastic items
    'bottle': 'plastic',
    'plastic bag': 'plastic',
    'plastic bottle': 'plastic',
    'water bottle': 'plastic',
    'soda bottle': 'plastic',
    'oil bottle': 'plastic',
    'vase': 'plastic',
    'cup': 'plastic',
    'plastic cup': 'plastic',
    'container': 'plastic',
    'jug': 'plastic',
    'jerry can': 'plastic',
    'plastic container': 'plastic',
    'straw': 'plastic',
    'plastic wrap': 'plastic',
    'packaging': 'plastic',
    'bucket': 'plastic',
    'detergent': 'plastic',
    
    # Metal items
    'can': 'metal',
    'tin can': 'metal',
    'aluminum can': 'metal',
    'soda can': 'metal',
    'fork': 'metal',
    'knife': 'metal',
    'spoon': 'metal',
    'metal lid': 'metal',
    'foil': 'metal',
    'aluminum foil': 'metal',
    
    # Paper items
    'book': 'paper',
    'box': 'paper',
    'paper': 'paper',
    'newspaper': 'paper',
    'magazine': 'paper',
    'cardboard': 'paper',
    'cardboard box': 'paper',
    'paper bag': 'paper',
    'carton': 'paper',
    'milk carton': 'paper',
    'juice carton': 'paper',
    
    # Glass items
    'wine glass': 'glass',
    'glass bottle': 'glass',
    'beer bottle': 'glass',
    'wine bottle': 'glass',
    'jar': 'glass',
    'glass jar': 'glass',
    
    # Organic waste
    'banana': 'organic',
    'apple': 'organic',
    'orange': 'organic',
    'carrot': 'organic',
    'food': 'organic',
    'fruit': 'organic',
    'vegetable': 'organic',
    'plant': 'organic',
    'leaf': 'organic',
    'leaves': 'organic',
    'coffee grounds': 'organic',
    'tea bag': 'organic',
    
    # Electronics
    'cell phone': 'electronic',
    'laptop': 'electronic',
    'mouse': 'electronic',
    'keyboard': 'electronic',
    'remote': 'electronic',
    'battery': 'electronic',
    'charger': 'electronic',
    'cable': 'electronic',
    'cord': 'electronic',
    'headphones': 'electronic',
    
    # Default to mixed if unrecognized
    'backpack': 'mixed',
    'umbrella': 'mixed',
    'handbag': 'mixed',
    'chair': 'mixed',
    'couch': 'mixed',
}

# ====================================================
# WATER BOTTLE DETECTION SPECIALIST - For common plastic bottles
# ====================================================

class WaterBottleDetector:
    """Specialized detector for common plastic bottles"""
    
    @staticmethod
    def is_water_bottle(img_crop, class_name):
        """
        Check if the object is likely a water/beverage bottle
        Returns (is_bottle, confidence)
        """
        # Convert to numpy array for analysis
        img_array = np.array(img_crop)
        
        # Quick check if class is already a good match
        if class_name.lower() in ['bottle', 'water bottle', 'plastic bottle']:
            # Calculate key features
            brightness = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2])
            avg_brightness = np.mean(brightness)
            brightness_std = np.std(brightness)
            
            # Check for brightness characteristics of clear plastic bottles
            if avg_brightness > 170 and brightness_std > 20:
                # Check for label area (usually a band in the middle)
                height, width = img_array.shape[0], img_array.shape[1]
                middle_section = img_array[int(height*0.3):int(height*0.7), :, :]
                
                # Check for color in the label area
                color_std = np.std(middle_section, axis=(0,1))
                if color_std.mean() > 40:  # More color variation in the label area
                    # It's likely a water/beverage bottle with a label
                    return True, 0.85
                elif avg_brightness > 200:
                    # It might be a clear water bottle without much label
                    return True, 0.75
            
        return False, 0.0

# ====================================================
# MAIN PREDICTION FUNCTION
# ====================================================

def predict_image(image_path):
    """
    Main function to predict objects and their materials in an image
    
    Parameters:
        image_path: Path to the image file
        
    Returns:
        List of predictions with class, material type, confidence, bbox, and volume
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # Get image dimensions
        height, width = img_np.shape[:2]
        
        # Make prediction with the object detection model
        results = MODEL(img)
        
        # Process results
        predictions = []
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            if float(conf) > 0.3:  # Confidence threshold
                x1, y1, x2, y2 = [float(x) for x in box]
                
                # Convert to normalized coordinates
                width_img, height_img = img.size
                x = x1 / width_img
                y = y1 / height_img
                w = (x2 - x1) / width_img
                h = (y2 - y1) / height_img
                
                # Extract the crop for material classification
                img_crop = img.crop((x1, y1, x2, y2))
                img_crop_np = np.array(img_crop)
                
                # Get the object class name
                class_id = int(cls)
                class_name = results.names[class_id] if hasattr(results, 'names') else f'class_{class_id}'
                
                # Enhanced shape analysis for improved accuracy
                shape_features = analyze_object_shape(img_crop_np)
                
                # Special case for water bottles - they're very common and often misclassified
                is_water_bottle, wb_confidence = WaterBottleDetector.is_water_bottle(img_crop, class_name)
                if is_water_bottle and wb_confidence > 0.7:
                    # Calculate adjusted bounding box based on shape analysis
                    adjusted_bbox = adjust_bbox_by_shape(shape_features, [x, y, w, h])
                    
                    # Calculate approximate volume with improved accuracy
                    volume = calculate_volume_from_shape(shape_features, width_img, height_img)
                    
                    predictions.append({
                        'class': 'plastic',
                        'detailed_class': 'plastic water bottle',
                        'confidence': wb_confidence,
                        'bbox': adjusted_bbox,
                        'estimated_volume': round(volume, 2),
                        'shape_info': shape_features['shape_type']
                    })
                    continue
                
                # For white containers like in the image
                # Direct detection for plastic containers (jerry cans, etc.)
                if w/h > 0.3 and w/h < 2.0:  # Reasonable aspect ratio for containers
                    img_array = np.array(img_crop)
                    avg_color = np.mean(img_array, axis=(0,1))
                    color_std = np.std(img_array, axis=(0,1))
                    
                    # Light colored, uniform plastic containers
                    if avg_color.mean() > 180 and color_std.mean() < 35:
                        # Adjust bounding box based on shape analysis
                        adjusted_bbox = adjust_bbox_by_shape(shape_features, [x, y, w, h])
                        
                        # Calculate improved volume estimate
                        volume = calculate_volume_from_shape(shape_features, width_img, height_img)
                        
                        predictions.append({
                            'class': 'plastic',
                            'detailed_class': 'plastic container',
                            'confidence': 0.85,
                            'bbox': adjusted_bbox,
                            'estimated_volume': round(volume, 2),
                            'shape_info': shape_features['shape_type']
                        })
                        continue
                
                # Advanced material classification
                material_type, confidence = MaterialAnalyzer.classify_material(
                    img_crop, class_name, shape_ratio=w/h
                )
                
                # If the confidence is too low, fall back to the mapping table
                if confidence < 0.6:
                    fallback_material = OBJECT_TO_WASTE_MAP.get(class_name.lower(), 'mixed')
                    if fallback_material != 'mixed':  # If we have a specific mapping
                        material_type = fallback_material
                        confidence = max(confidence, 0.7)  # Boost confidence for known mappings
                
                # Generate a detailed class description
                if material_type in ['plastic', 'glass', 'metal', 'paper']:
                    detailed_class = f"{material_type} {class_name.lower()}"
                else:
                    detailed_class = class_name
                
                # Calculate standard volume for regular objects
                pixel_to_real_world_ratio = 0.01  # This is a placeholder value
                real_width = w * width_img * pixel_to_real_world_ratio
                real_height = h * height_img * pixel_to_real_world_ratio
                estimated_depth = real_width * 0.7
                estimated_volume = real_width * real_height * estimated_depth
                
                # Standard prediction
                predictions.append({
                    'class': material_type,
                    'detailed_class': detailed_class,
                    'confidence': float(confidence) * float(conf),  # Combine confidences
                    'bbox': [x, y, w, h],
                    'estimated_volume': round(estimated_volume, 2)
                })
        
        return predictions
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

def analyze_object_shape(image_np):
    """Analyze the shape characteristics of an object in the cropped image"""
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = np.mean(image_np, axis=2).astype(np.uint8)
    else:
        gray = image_np.astype(np.uint8)
    
    # Basic shape features
    height, width = gray.shape[:2]
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Intensity distribution - helps identify transparent vs opaque objects
    hist = np.histogram(gray, bins=64, range=(0, 256))[0]
    hist_normalized = hist / hist.sum() if hist.sum() > 0 else hist
    
    # Try to determine if the object is cylindrical (like a bottle)
    # Cylindrical objects often have symmetric gradients from center to edges
    
    # Compute horizontal and vertical gradients
    h_center = width // 2
    v_center = height // 2
    
    # Compute distances from center
    y_indices, x_indices = np.indices((height, width))
    distances = np.sqrt((x_indices - h_center) ** 2 + (y_indices - v_center) ** 2)
    max_distance = np.max(distances)
    normalized_distances = distances / max_distance if max_distance > 0 else distances
    
    # Simple shape classifier
    if aspect_ratio > 2.0:
        shape_type = "tall_cylindrical"  # Likely a bottle or tall container
    elif aspect_ratio < 0.5:
        shape_type = "flat_rectangular"  # Likely a flat, wide container
    elif aspect_ratio > 0.8 and aspect_ratio < 1.2:
        shape_type = "square_or_round"   # Could be a round container or square box
    else:
        shape_type = "rectangular"       # Standard rectangular object
    
    # Calculate brightness gradient from center
    if shape_type == "tall_cylindrical" or shape_type == "square_or_round":
        # Cylindrical objects often have reflections that create brightness gradients
        center_brightness = np.mean(gray[v_center-5:v_center+5, h_center-5:h_center+5])
        edge_brightness = np.mean([
            np.mean(gray[0:10, :]),  # top
            np.mean(gray[-10:, :]),  # bottom
            np.mean(gray[:, 0:10]),  # left
            np.mean(gray[:, -10:])   # right
        ])
        brightness_gradient = center_brightness - edge_brightness
        
        # Strong gradient often indicates a rounded object
        if brightness_gradient > 30:
            shape_type = "cylindrical" if aspect_ratio > 1.2 else "round"
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "shape_type": shape_type,
        "histogram": hist_normalized.tolist(),
        "brightness_gradient": brightness_gradient if 'brightness_gradient' in locals() else 0
    }

def adjust_bbox_by_shape(shape_features, original_bbox):
    """Fine-tune bounding box based on shape analysis"""
    x, y, w, h = original_bbox
    
    # Don't adjust too much, just slight tweaks based on shape type
    if shape_features["shape_type"] == "cylindrical" or shape_features["shape_type"] == "tall_cylindrical":
        # Make cylindrical objects slightly narrower and taller
        w_adjust = w * 0.05
        return [x + w_adjust, y, w - (w_adjust * 2), h]
    elif shape_features["shape_type"] == "round":
        # Make round objects more square
        if w > h:
            diff = (w - h) / 2
            return [x + diff, y, h, h]  # x, y, w, h format
        else:
            diff = (h - w) / 2
            return [x, y + diff, w, w]  # x, y, w, h format
    
    # Default: return original bbox
    return original_bbox

def calculate_volume_from_shape(shape_features, img_width, img_height):
    """Calculate a more accurate volume estimation based on shape analysis"""
    shape_type = shape_features["shape_type"]
    width = shape_features["width"] 
    height = shape_features["height"]
    
    # Base ratio for converting pixels to real-world units
    pixel_to_real_world_ratio = 0.01
    
    # Convert to real-world dimensions
    real_width = width * pixel_to_real_world_ratio
    real_height = height * pixel_to_real_world_ratio
    
    # Volume calculation based on shape type
    if shape_type == "cylindrical" or shape_type == "tall_cylindrical":
        # Use cylinder formula: π * r² * h
        radius = real_width / 2
        volume = math.pi * (radius ** 2) * real_height
    elif shape_type == "round":
        # Use sphere formula: (4/3) * π * r³
        radius = max(real_width, real_height) / 2
        volume = (4/3) * math.pi * (radius ** 3)
    else:
        # Use rectangular prism formula: w * h * d
        # Assume depth is 70% of width for standard objects
        estimated_depth = real_width * 0.7
        volume = real_width * real_height * estimated_depth
    
    return volume

# ====================================================
# COMMAND LINE INTERFACE
# ====================================================

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Image path not provided'}), file=sys.stderr)
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({'error': f'Image not found: {image_path}'}), file=sys.stderr)
        sys.exit(1)
        
    try:
        results = predict_image(image_path)
        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)