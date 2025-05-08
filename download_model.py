import os
import bz2
import urllib.request

def download_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download the shape predictor model if it doesn't exist
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("Downloading shape predictor model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, predictor_path + ".bz2")
        
        print("Extracting model file...")
        with bz2.open(predictor_path + ".bz2", 'rb') as source, open(predictor_path, 'wb') as dest:
            dest.write(source.read())
        
        os.remove(predictor_path + ".bz2")
        print("Model downloaded and extracted successfully!")
    else:
        print("Model file already exists.")

if __name__ == '__main__':
    download_model() 