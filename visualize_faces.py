import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def get_gender_color(gender):
    """Return color based on gender"""
    if gender.lower() == 'man':
        return (1, 0, 0)  # Red for men
    elif gender.lower() == 'woman':
        return (0, 0, 1)  # Blue for women
    return (0, 1, 0)  # Green for unknown

def find_image_by_id(image_id):
    """Find image file that corresponds to the given image_id"""
    # Search in common image directories
    search_dirs = ['data', 'images', 'input', '.']
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        # Search for image files
        for ext in ['jpg', 'jpeg', 'png']:
            pattern = os.path.join(directory, f"*{image_id}*.{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
    return None

def visualize_faces(json_path, output_dir='output'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image ID and original filename
    image_id = data.get('image_id', '')
    original_filename = data.get('original_filename', '')
    
    # Find corresponding image file
    image_file = None
    if original_filename:
        # Search in common image directories
        search_dirs = ['data', 'images', 'input', '.']
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
            potential_path = os.path.join(directory, original_filename)
            if os.path.exists(potential_path):
                image_file = potential_path
                break
    
    if image_file is None:
        print(f"Warning: Could not find original image file '{original_filename}', using default image")
        image_file = "data/MultipleFaces.jpg"
    
    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not read image {image_file}")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # Process each face
    for face in data.get('faces', []):
        face_id = face.get('face_id', '')
        bbox = face.get('bbox', {})
        
        # Get face coordinates
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        # Get age/gender information
        age_gender = face.get('age_gender', {})
        age = age_gender.get('age', 'N/A')
        gender = age_gender.get('gender', 'N/A')
        confidence = age_gender.get('confidence', 0)
        
        # Get color based on gender
        color = get_gender_color(gender)
        
        # Draw bounding box
        rect = plt.Rectangle((x, y), width, height, 
                           fill=False, 
                           edgecolor=color, 
                           linewidth=2)
        plt.gca().add_patch(rect)
        
        # Draw landmarks if available
        if 'landmarks' in face:
            landmarks = face['landmarks']
            if 'all_points' in landmarks:
                points = np.array(landmarks['all_points'])
                plt.plot(points[:, 0], points[:, 1], '.', 
                        color=color, 
                        markersize=2, 
                        alpha=0.5)
        
        # Add information text
        info_text = f"Face ID: {face_id[:8]}...\nAge: {age}\nGender: {gender}\nConfidence: {confidence:.2f}%"
        
        # Add text with background
        plt.text(x, y-10, info_text,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7),
                verticalalignment='top')
    
    # Remove axis
    plt.axis('off')
    
    # Add title with image ID and timestamp
    timestamp = data.get('timestamp', 'N/A')
    if timestamp != 'N/A':
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
    
    plt.title(f"Image ID: {image_id}\nTimestamp: {timestamp}\nOriginal File: {original_filename}",
              fontsize=12, pad=20)
    
    # Save the figure
    output_file = os.path.join(output_dir, f"{image_id}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"Saved visualization to {output_file}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the figure to free memory

def main():
    # Get the storage directory path
    storage_dir = 'storage'
    
    # Check if storage directory exists
    if not os.path.exists(storage_dir):
        print(f"Storage directory '{storage_dir}' not found!")
        return
    
    # Get all JSON files in the storage directory
    json_files = [f for f in os.listdir(storage_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in storage directory!")
        return
    
    # Process each JSON file
    for json_file in json_files:
        print(f"\nProcessing {json_file}...")
        visualize_faces(os.path.join(storage_dir, json_file))

if __name__ == "__main__":
    main() 