import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import dlib
import json
import time
import base64
import logging
import numpy as np
from concurrent import futures
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc
import grpc
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
FACE_LANDMARK_PORT = 50051

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class FaceLandmarkService(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self):
        # Initialize dlib's facial landmarks predictor
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        logger.info("Loading dlib models...")
        logger.info("dlib models loaded successfully")

    def extract_face_region(self, image, landmarks):
        """Extract face region based on landmarks with padding"""
        points = np.array(landmarks['all_points'])
        x, y = points[:, 0].min(), points[:, 1].min()
        w = points[:, 0].max() - x
        h = points[:, 1].max() - y
        
        # Add padding (20%)
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        
        # Calculate new coordinates with padding
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        
        # Encode face region as base64
        _, buffer = cv2.imencode('.jpg', face_region)
        face_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'face_image': face_image_base64,
            'bbox': {
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1)
            }
        }

    def SaveFaceAttributes(self, request, context):
        start_time = time.time()
        logger.info(f"Processing face landmarks for image {request.image_id}")
        
        # Convert frame data to image
        nparr = np.frombuffer(request.frame, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process faces
        faces = []
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for face_info in request.faces:
                try:
                    face_id = face_info.face_id
                    bbox = face_info.bbox
                    
                    # Create dlib rectangle from bbox
                    rect = dlib.rectangle(
                        bbox.x,
                        bbox.y,
                        bbox.x + bbox.width,
                        bbox.y + bbox.height
                    )
                    
                    # Get facial landmarks
                    shape = self.predictor(gray, rect)
                    landmarks = []
                    
                    # Convert landmarks to list of (x,y) coordinates
                    for i in range(68):
                        x = shape.part(i).x
                        y = shape.part(i).y
                        landmarks.append([x, y])
                    
                    # Create landmarks dictionary
                    landmarks_dict = {
                        'left_eye': landmarks[36],  # Left eye corner
                        'right_eye': landmarks[45],  # Right eye corner
                        'nose': landmarks[30],      # Nose tip
                        'mouth_left': landmarks[48], # Left mouth corner
                        'mouth_right': landmarks[54], # Right mouth corner
                        'all_points': landmarks
                    }
                    
                    # Extract face region and encode it
                    face_data = self.extract_face_region(image, landmarks_dict)
                    
                    # Combine all face data
                    face_info = {
                        'face_id': face_id,
                        'landmarks': landmarks_dict,
                        'face_image': face_data['face_image'],
                        'bbox': face_data['bbox'],
                        'confidence': 0.95
                    }
                    
                    faces.append(face_info)
                    logger.info(f"Detected landmarks for face {face_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to detect landmarks for face {face_id}")
                    logger.warning(str(e))
                    continue
            
        except Exception as e:
            logger.warning("Error processing faces")
            logger.warning(str(e))
        
        # Store results in Redis
        if faces:
            self.redis_client.set(f"face_landmarks:{request.image_id}", json.dumps(faces))
        
        processing_time = time.time() - start_time
        logger.info(f"Face landmark processing completed in {processing_time:.2f} seconds")
        
        return aggregator_pb2.FaceResultResponse(
            response=True,
            message=f"Processed {len(faces)} faces"
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(FaceLandmarkService(), server)
    server.add_insecure_port(f'[::]:{FACE_LANDMARK_PORT}')
    server.start()
    logger.info(f"Face Landmark Service started on port {FACE_LANDMARK_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 