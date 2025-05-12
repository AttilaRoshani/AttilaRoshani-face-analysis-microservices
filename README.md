# Face Analysis Microservices 🎯

<div align="center">
  <img src="output/2e99c145-f75b-4015-852b-c809d6452484.png" alt="Face Analysis Result" width="600"/>
  <br>
  <em>Sample output of face analysis with age-gender and  facial feature detection</em>
</div>

## 📝 Description
This project is a microservices-based face analysis system that provides the following capabilities:
- Face Detection
- Age and Gender Estimation
- Facial Feature Detection
- Result Storage and Retrieval

## 🚀 Features
- Microservices Architecture using FastAPI
- Image Processing with OpenCV
- Face Detection with DeepFace
- Redis Storage Integration
- Web Interface with Streamlit
- Docker Support

## 🛠️ Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Redis

## 📦 Installation

### Using Docker
```bash
# Clone the repository
git clone https://github.com/AttilaRoshani/face-analysis-microservices.git
cd face-analysis-microservices

# Run with Docker Compose
docker-compose up --build
```

### Manual Installation
```bash
# Make the script executable
chmod +x start_services.sh

# Run services
./start_services.sh
```

## 🎮 Usage

### API Endpoints
- `POST /analyze`: Analyze an image
- `GET /results/{id}`: Get analysis results
- `GET /health`: Check service status

### Example Request
```python
import requests

# Upload and analyze image
response = requests.post(
    "http://localhost:8000/analyze",
    files={"file": open("image.jpg", "rb")}
)

# Get results
result_id = response.json()["id"]
results = requests.get(f"http://localhost:8000/results/{result_id}")
```

## 📊 Project Structure
```
face-analysis-microservices/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   └── services/
├── storage/
├── data/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🤝 Contributing
We welcome contributions to improve this project. To contribute:
1. Create an Issue
2. Submit a Pull Request
3. Follow the coding guidelines

## 📄 License
This project is licensed under the MIT License.

## 👥 Authors
- [Attila Roshani](https://github.com/AttilaRoshani)

---
<div align="center">
  <sub>Built with ❤️ by Attila Roshani</sub>
</div> 
