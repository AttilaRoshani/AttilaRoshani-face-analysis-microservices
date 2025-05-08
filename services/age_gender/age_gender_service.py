import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import grpc
import time
import logging
import redis
import json
import cv2
import numpy as np
from concurrent import futures
from deepface import DeepFace
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
AGE_GENDER_PORT = 50052

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class AgeGenderService(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        # Initialize DeepFace models
        logger.info("Loading DeepFace models...")
        try:
            # Analyze a dummy image to load models into memory
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['age', 'gender'], enforce_detection=False)
            logger.info("DeepFace models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading DeepFace models: {e}")

    def analyze_face(self, face_image):
        """Analyze a single face image using DeepFace"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(face_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Analyze face
            result = DeepFace.analyze(
                img, 
                actions=['age', 'gender'],
                enforce_detection=False,  # Skip face detection as we already have face crops
                silent=True
            )

            if isinstance(result, list):
                result = result[0]  # Take first result if multiple faces detected

            # Extract gender information
            gender = result.get("dominant_gender", "unknown").lower()
            gender_prob = result.get("gender", {})
            if isinstance(gender_prob, dict):
                confidence = max(gender_prob.values()) if gender_prob else 0.0
            else:
                confidence = 0.0

            return {
                "age": int(result.get("age", 25)),
                "gender": gender,
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Error analyzing face: {e}")
            # Return default values if analysis fails
            return {
                "age": 25,
                "gender": "unknown",
                "confidence": 0.0
            }

    def SaveFaceAttributes(self, request, context):
        start_time = time.time()
        logger.info(f"Processing age/gender for image {request.image_id}")

        # Process each face in the image
        face_results = []
        for face in request.faces:
            # Analyze face using DeepFace
            age_gender = self.analyze_face(face.face_image)

            face_result = {
                "face_id": face.face_id,
                "age_gender": age_gender
            }
            face_results.append(face_result)

            logger.info(f"Face {face.face_id}: Age={age_gender['age']}, Gender={age_gender['gender']}")

        # Store results in Redis
        redis_key = f"age_gender:{request.image_id}"
        self.redis_client.set(redis_key, json.dumps(face_results))

        processing_time = time.time() - start_time
        logger.info(f"Age/gender processing completed in {processing_time:.2f} seconds")

        return aggregator_pb2.FaceResultResponse(
            response=True,
            message=f"Processed age/gender for {len(face_results)} faces in image {request.image_id}"
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(AgeGenderService(), server)
    server.add_insecure_port(f'[::]:{AGE_GENDER_PORT}')
    server.start()
    logger.info(f"Age/Gender Service started on port {AGE_GENDER_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 