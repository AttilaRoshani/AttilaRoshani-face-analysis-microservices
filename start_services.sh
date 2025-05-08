#!/bin/bash

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start Redis if not running
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis server..."
    redis-server &
    sleep 2
fi

# Start all services
echo "Starting Image Input Service..."
python services/image_input/image_input_service.py &
sleep 2

echo "Starting Face Landmark Service..."
python services/face_landmark/face_landmark_service.py &
sleep 2

echo "Starting Age/Gender Service..."
python services/age_gender/age_gender_service.py &
sleep 2

echo "Starting Data Storage Service..."
python services/data_storage/data_storage_service.py &
sleep 2

echo "All services started successfully!"

# Wait for services to be fully initialized
echo "Waiting for services to initialize..."
sleep 5

# Run main.py
echo "Starting main processing..."
python main.py

# Keep the script running to maintain services
echo "Press Ctrl+C to stop all services"
wait 