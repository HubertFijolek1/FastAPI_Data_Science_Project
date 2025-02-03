from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
import cv2
import os

from app import data_processing, predictive_analysis, image_video_recognition

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, FastAPI World!"}

@app.post("/data/analysis")
async def analyze_data(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV and return basic stats.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df = data_processing.load_and_clean_data_from_df(df)
    stats = data_processing.analyze_data(df)
    return stats.to_dict()

@app.post("/predict")
async def predict_endpoint(features: dict):
    """
    Trains a linear model on data.csv (assuming 'feature1','feature2','target'),
    then predicts for incoming JSON features.
    """
    df = pd.read_csv("data.csv")  # Example CSV
    model = predictive_analysis.train_predictive_model(df, ["feature1", "feature2"], "target")
    new_data = pd.DataFrame([features])
    prediction = predictive_analysis.predict(model, new_data)
    return {"prediction": prediction.tolist()}

@app.post("/faces/detect")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    """
    Upload an image, detect faces using OpenCV.
    """
    temp_file = "temp_image.jpg"
    try:
        contents = await file.read()
        with open(temp_file, "wb") as f:
            f.write(contents)
        faces = image_video_recognition.detect_faces(temp_file)
        return {"num_faces": len(faces), "faces": faces.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)