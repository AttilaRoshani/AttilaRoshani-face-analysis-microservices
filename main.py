import os
import time
import grpc
import logging
import subprocess
import threading
import uuid
from pathlib import Path
import cv2
import numpy as np
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_service(service_name, service_path):
    """Start a service in a separate process"""
    logger.info(f"Starting {service_name}...")
    process = subprocess.Popen(['python', service_path])
    time.sleep(2)  # Wait for service to start
    return process

def detect_faces(image_data):
    """Detect faces in the image using OpenCV"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_data_list = []
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = img[y:y+h, x:x+w]
        # Convert face image to bytes
        _, face_bytes = cv2.imencode('.jpg', face_img)
        
        face_data = aggregator_pb2.FaceData(
            face_id=str(uuid.uuid4()),
            face_image=face_bytes.tobytes()
        )
        face_data_list.append(face_data)
    
    return face_data_list

def process_image(image_path, stub):
    """Process a single image through the image input service"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create request
        request = aggregator_pb2.FaceResult(
            time=str(time.time()),
            frame=image_data,
            image_id=str(uuid.uuid4()),
            original_filename=os.path.basename(image_path)
        )
        
        # Send request
        response = stub.SaveFaceAttributes(request)
        logger.info(f"Processed {image_path}: {response.message}")
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Start all services
    services = [
        ('Data Storage Service', 'services/data_storage/data_storage_service.py'),
        ('Face Landmark Service', 'services/face_landmark/face_landmark_service.py'),
        ('Age/Gender Service', 'services/age_gender/age_gender_service.py'),
        ('Image Input Service', 'services/image_input/image_input_service.py')
    ]
    
    processes = []
    for service_name, service_path in services:
        process = start_service(service_name, service_path)
        processes.append(process)
    
    try:
        # Wait for services to start
        time.sleep(5)
        
        # Connect to image input service
        channel = grpc.insecure_channel('localhost:50050')
        stub = aggregator_pb2_grpc.AggregatorStub(channel)
        
        # Process images from data directory
        data_dir = Path('data')
        if not data_dir.exists():
            logger.error("Data directory not found!")
            return
        
        image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
        if not image_files:
            logger.error("No images found in data directory!")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_path in image_files:
            logger.info(f"Processing {image_path}")
            process_image(str(image_path), stub)
            time.sleep(1)  # Small delay between images
        
        # Wait for processing to complete
        time.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Stop all services
        for process in processes:
            process.terminate()
        
        # Wait for processes to terminate
        for process in processes:
            process.wait()

if __name__ == '__main__':
    main() 