import os
import grpc
import time
import logging
import uuid
from datetime import datetime
import aggregator_pb2
import aggregator_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
IMAGE_INPUT_PORT = 50050

class FaceDetectionClient:
    def __init__(self):
        self.channel = grpc.insecure_channel(f'localhost:{IMAGE_INPUT_PORT}')
        self.stub = aggregator_pb2_grpc.AggregatorStub(self.channel)

    def send_test_data(self):
        """Send test data to the server"""
        # Create dummy image data (1x1 pixel black image)
        dummy_image = b'\x00' * 100  # Small dummy image data

        # Create face data (dummy data for testing)
        faces = []
        # Example: Create 2 dummy faces
        for i in range(2):
            face = aggregator_pb2.FaceData(
                face_id=str(uuid.uuid4()),
                face_image=dummy_image,
                landmarks=aggregator_pb2.FaceLandmarks(
                    points=[
                        aggregator_pb2.Point(x=100 + i*100, y=100),
                        aggregator_pb2.Point(x=150 + i*100, y=100),
                        aggregator_pb2.Point(x=125 + i*100, y=150)
                    ]
                ),
                age_gender=aggregator_pb2.AgeGenderInfo(
                    age=25 + i*5,
                    gender="male" if i == 0 else "female",
                    confidence=0.95
                )
            )
            faces.append(face)

        # Create request
        request = aggregator_pb2.FaceResult(
            time=datetime.now().isoformat(),
            frame=dummy_image,
            image_id=str(uuid.uuid4()),
            faces=faces
        )

        try:
            # Send request to server
            response = self.stub.SaveFaceAttributes(request)
            logger.info(f"Server response: {response.message}")
            return response
        except grpc.RpcError as e:
            logger.error(f"RPC failed: {e}")
            raise

def main():
    # Create client
    client = FaceDetectionClient()

    try:
        # Send test data
        response = client.send_test_data()
        logger.info("Test data sent successfully")
    except Exception as e:
        logger.error(f"Error sending test data: {e}")

if __name__ == '__main__':
    main() 