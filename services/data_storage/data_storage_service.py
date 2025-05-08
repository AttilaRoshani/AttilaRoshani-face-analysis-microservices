import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
import logging
import redis
from concurrent import futures
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc
import grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
DATA_STORAGE_PORT = 50053

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class DataStorageService(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        # Create storage directory if it doesn't exist
        os.makedirs('storage', exist_ok=True)

    def SaveFaceAttributes(self, request, context):
        start_time = time.time()
        logger.info(f"Storing data for image {request.image_id}")

        try:
            # Get face landmarks data
            landmarks_key = f"face_landmarks:{request.image_id}"
            landmarks_data = self.redis_client.get(landmarks_key)
            logger.info(f"Retrieved landmarks data: {landmarks_data}")
            
            # Get age/gender data
            age_gender_key = f"age_gender:{request.image_id}"
            age_gender_data = self.redis_client.get(age_gender_key)
            logger.info(f"Retrieved age/gender data: {age_gender_data}")

            # Get original image filename from Redis
            image_filename_key = f"image_filename:{request.image_id}"
            original_filename = self.redis_client.get(image_filename_key)
            if not original_filename:
                original_filename = "unknown.jpg"  # Default if not found

            # If either data is missing, store what we have and return
            if not landmarks_data or not age_gender_data:
                logger.warning(f"Missing data for image {request.image_id}")
                if not landmarks_data:
                    logger.warning("No landmarks data found")
                if not age_gender_data:
                    logger.warning("No age/gender data found")
                
                # Store partial data if available
                partial_data = []
                if landmarks_data:
                    landmarks = json.loads(landmarks_data)
                    for landmark_item in landmarks:
                        partial_data.append({
                            'face_id': landmark_item['face_id'],
                            'landmarks': landmark_item['landmarks'],
                            'face_image': landmark_item['face_image'],
                            'bbox': landmark_item['bbox'],
                            'timestamp': request.time
                        })
                elif age_gender_data:
                    age_gender = json.loads(age_gender_data)
                    for item in age_gender:
                        partial_data.append({
                            'face_id': item['face_id'],
                            'age_gender': item['age_gender'],
                            'timestamp': request.time
                        })

                if partial_data:
                    # Store partial data in Redis
                    storage_key = f"face_data:{request.image_id}"
                    self.redis_client.set(storage_key, json.dumps(partial_data))

                    # Save to file in storage directory using original filename
                    base_filename = os.path.splitext(original_filename)[0]
                    storage_file = os.path.join('storage', f"{base_filename}.json")
                    with open(storage_file, 'w') as f:
                        json.dump({
                            'image_id': request.image_id,
                            'original_filename': original_filename,
                            'timestamp': request.time,
                            'faces': partial_data
                        }, f, indent=2)

                    logger.info(f"Stored partial data for {len(partial_data)} faces")
                    return aggregator_pb2.FaceResultResponse(
                        response=True,
                        message=f"Stored partial data for {len(partial_data)} faces in image {request.image_id}"
                    )

                return aggregator_pb2.FaceResultResponse(
                    response=False,
                    message="No data available for storage"
                )

            # Combine all data
            landmarks = json.loads(landmarks_data)
            age_gender = json.loads(age_gender_data)

            # Create a map of face_id to age/gender data
            age_gender_map = {item['face_id']: item['age_gender'] for item in age_gender}

            # Combine data for each face
            combined_data = []
            for landmark_item in landmarks:
                face_id = landmark_item['face_id']
                if face_id in age_gender_map:
                    combined_data.append({
                        'face_id': face_id,
                        'landmarks': landmark_item['landmarks'],
                        'face_image': landmark_item['face_image'],
                        'bbox': landmark_item['bbox'],
                        'age_gender': age_gender_map[face_id],
                        'timestamp': request.time
                    })
                else:
                    # If age/gender data is missing for this face, still include the landmark data
                    combined_data.append({
                        'face_id': face_id,
                        'landmarks': landmark_item['landmarks'],
                        'face_image': landmark_item['face_image'],
                        'bbox': landmark_item['bbox'],
                        'timestamp': request.time
                    })

            # Add any faces that only have age/gender data
            for item in age_gender:
                face_id = item['face_id']
                if not any(face['face_id'] == face_id for face in combined_data):
                    combined_data.append({
                        'face_id': face_id,
                        'age_gender': item['age_gender'],
                        'timestamp': request.time
                    })

            # Store combined data in Redis
            storage_key = f"face_data:{request.image_id}"
            self.redis_client.set(storage_key, json.dumps(combined_data))

            # Save to file in storage directory using original filename
            base_filename = os.path.splitext(original_filename)[0]
            storage_file = os.path.join('storage', f"{base_filename}.json")
            with open(storage_file, 'w') as f:
                json.dump({
                    'image_id': request.image_id,
                    'original_filename': original_filename,
                    'timestamp': request.time,
                    'faces': combined_data
                }, f, indent=2)

            processing_time = time.time() - start_time
            logger.info(f"Data storage completed in {processing_time:.2f} seconds")
            logger.info(f"Results saved to {storage_file}")

            return aggregator_pb2.FaceResultResponse(
                response=True,
                message=f"Stored data for {len(combined_data)} faces in image {request.image_id}"
            )
        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
            return aggregator_pb2.FaceResultResponse(
                response=False,
                message=f"Error storing data: {str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(DataStorageService(), server)
    server.add_insecure_port(f'[::]:{DATA_STORAGE_PORT}')
    server.start()
    logger.info(f"Data Storage Service started on port {DATA_STORAGE_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 