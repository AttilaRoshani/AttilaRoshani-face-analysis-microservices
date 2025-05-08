# Face Analysis Microservices System

A robust microservices-based system for face detection, landmark detection, and age/gender estimation using Docker containers.

## Features

- **Face Detection**: Detects multiple faces in images using OpenCV
- **Landmark Detection**: Extracts 68 facial landmarks using dlib
- **Age/Gender Estimation**: Estimates age and gender using DeepFace
- **Microservices Architecture**: Each component runs as a separate service
- **Docker Containerization**: Easy deployment and scaling
- **Redis Integration**: Fast data storage and communication
- **gRPC Communication**: Efficient service-to-service communication
- **Visualization Tools**: View results with bounding boxes and landmarks

## System Architecture

The system consists of four microservices:

1. **Image Input Service** (Port 50050)
   - Receives and preprocesses images
   - Distributes images to other services
   - Manages image flow through the pipeline

2. **Face Landmark Service** (Port 50051)
   - Detects faces using dlib
   - Extracts 68 facial landmarks
   - Provides face bounding boxes

3. **Age/Gender Service** (Port 50052)
   - Estimates age using DeepFace
   - Determines gender using DeepFace
   - Provides confidence scores

4. **Data Storage Service** (Port 50053)
   - Stores results in Redis
   - Saves JSON files with all face data
   - Manages data persistence

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Redis (included in Docker setup)
- Internet connection for initial model downloads

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face-analysis-microservices
   ```

2. Download required models:
   ```bash
   python download_model.py
   ```

3. Generate gRPC code:
   ```bash
   python generate_grpc.py
   ```

4. Build and run with Docker:
   ```bash
   docker-compose up --build
   ```

## Usage

### 1. Preparing Input Images

- Place your images in the `data` directory
- Supported formats: JPG, PNG
- Multiple faces per image are supported

### 2. Running the System

Two methods to run the system:

#### Method 1: Using Docker (Recommended)
```bash
# Start all services
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

#### Method 2: Using start_services.sh
```bash
# Make script executable
chmod +x start_services.sh

# Run services
./start_services.sh
```

### 3. Viewing Results

Results are stored in two places:

1. **Redis Storage**:
   - Temporary storage during processing
   - Accessible by all services
   - Cleared on system restart

2. **JSON Files** (in `storage` directory):
   - Permanent storage
   - One JSON file per processed image
   - Contains all face data including:
     - Face landmarks
     - Age and gender estimates
     - Bounding boxes
     - Confidence scores

### 4. Visualizing Results

To visualize the results:
```bash
python visualize_faces.py
```

This will:
- Read JSON files from the `storage` directory
- Create visualizations with:
  - Face bounding boxes
  - Facial landmarks
  - Age and gender information
- Save visualizations in the `output` directory

## Project Structure

```
project/
├── data/               # Input images
├── storage/           # Output JSON files
├── output/            # Visualization results
├── services/          # Microservices
│   ├── image_input/
│   ├── face_landmark/
│   ├── age_gender/
│   └── data_storage/
├── models/            # ML models
├── protos/           # gRPC protocol files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Development

### Running Individual Services

```bash
# Start Redis
docker-compose up redis

# Start specific service
docker-compose up <service-name>
```

### Testing

A test client is provided in `client/test_client.py`:
```bash
python client/test_client.py
```

### Adding New Features

1. Create new service in `services` directory
2. Add service to `docker-compose.yml`
3. Update gRPC protocol if needed
4. Rebuild and test

## Troubleshooting

1. **Service Connection Issues**:
   - Check if Redis is running
   - Verify service ports are available
   - Check Docker logs

2. **Model Loading Issues**:
   - Run `download_model.py` again
   - Check internet connection
   - Verify model files exist

3. **Docker Issues**:
   - Clear Docker cache: `docker system prune`
   - Rebuild containers: `docker-compose up --build`
   - Check Docker logs: `docker-compose logs`

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 