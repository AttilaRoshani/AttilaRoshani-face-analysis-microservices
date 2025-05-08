import os
import sys
import uuid
import time
import logging
import redis
import cv2
import numpy as np
import grpc
from concurrent import futures
from dotenv import load_dotenv
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
IMAGE_INPUT_PORT = 50050
FACE_LANDMARK_PORT = 50051
AGE_GENDER_PORT = 50052
DATA_STORAGE_PORT = 50053

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class ImageInputService(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self):
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        
        # Initialize face detector
        logger.info("Loading dlib face detector...")
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        logger.info("dlib face detector loaded successfully")
        
        # Initialize gRPC stubs
        self.face_landmark_channel = grpc.insecure_channel(f'localhost:{FACE_LANDMARK_PORT}')
        self.face_landmark_stub = aggregator_pb2_grpc.AggregatorStub(self.face_landmark_channel)
        
        self.age_gender_channel = grpc.insecure_channel(f'localhost:{AGE_GENDER_PORT}')
        self.age_gender_stub = aggregator_pb2_grpc.AggregatorStub(self.age_gender_channel)
        
        self.data_storage_channel = grpc.insecure_channel(f'localhost:{DATA_STORAGE_PORT}')
        self.data_storage_stub = aggregator_pb2_grpc.AggregatorStub(self.data_storage_channel)

    def SaveFaceAttributes(self, request, context):
        start_time = time.time()
        logger.info(f"Processing image {request.image_id}")

        # Store original image filename in Redis
        if hasattr(request, 'original_filename'):
            image_filename_key = f"image_filename:{request.image_id}"
            self.redis_client.set(image_filename_key, request.original_filename)

        # Convert frame data to image
        nparr = np.frombuffer(request.frame, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.detector(gray)

        # Process each detected face
        faces = []
        for face in detected_faces:
            # Generate unique ID for this face
            face_id = str(uuid.uuid4())
            
            # Get face coordinates
            x = face.left()
            y = face.top()
            width = face.right() - x
            height = face.bottom() - y
            
            # Extract face region
            face_img = image[y:y+height, x:x+width]
            _, face_bytes = cv2.imencode('.jpg', face_img)
            
            # Create face data
            face_data = aggregator_pb2.FaceData(
                face_id=face_id,
                face_image=face_bytes.tobytes(),
                bbox=aggregator_pb2.BoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height
                )
            )
            faces.append(face_data)
            logger.info(f"Detected face {face_id}")

        # Create new request with faces
        new_request = aggregator_pb2.FaceResult(
            time=request.time,
            frame=request.frame,
            image_id=request.image_id,
            faces=faces
        )

        # Send to face landmark service
        try:
            landmark_response = self.face_landmark_stub.SaveFaceAttributes(new_request)
            logger.info(f"Face landmark service response: {landmark_response.message}")
        except Exception as e:
            logger.error(f"Error calling face landmark service: {str(e)}")
            return aggregator_pb2.FaceResultResponse(
                response=False,
                message=f"Error calling face landmark service: {str(e)}"
            )

        # Send to age/gender service
        try:
            age_gender_response = self.age_gender_stub.SaveFaceAttributes(new_request)
            logger.info(f"Age/gender service response: {age_gender_response.message}")
        except Exception as e:
            logger.error(f"Error calling age/gender service: {str(e)}")
            return aggregator_pb2.FaceResultResponse(
                response=False,
                message=f"Error calling age/gender service: {str(e)}"
            )

        # Send to data storage service
        try:
            storage_response = self.data_storage_stub.SaveFaceAttributes(new_request)
            logger.info(f"Data storage service response: {storage_response.message}")
        except Exception as e:
            logger.error(f"Error calling data storage service: {str(e)}")
            return aggregator_pb2.FaceResultResponse(
                response=False,
                message=f"Error calling data storage service: {str(e)}"
            )

        processing_time = time.time() - start_time
        logger.info(f"Face detection completed in {processing_time:.2f} seconds")

        return aggregator_pb2.FaceResultResponse(
            response=True,
            message=f"Detected {len(faces)} faces in image {request.image_id}",
            faces=faces
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(ImageInputService(), server)
    server.add_insecure_port(f'[::]:{IMAGE_INPUT_PORT}')
    server.start()
    logger.info(f"Image Input Service started on port {IMAGE_INPUT_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 