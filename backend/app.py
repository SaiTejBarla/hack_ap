from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
from PIL import Image
import io
import logging

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

from crop_advisor import get_advice, normalize_disease_label

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & processor once
print("[INFO] Loading model...")
processor = AutoImageProcessor.from_pretrained(
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)
model = AutoModelForImageClassification.from_pretrained(
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

with open("labels.json") as f:
    labels = json.load(f)

print("[INFO] Model loaded successfully.")

CONFIDENCE_THRESHOLD = 0.80  # 80% confidence required

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_idx = torch.max(predictions, dim=1)

        label = model.config.id2label[predicted_idx.item()]
        normalized_label = normalize_disease_label(label)

        # Check confidence threshold
        if confidence.item() < CONFIDENCE_THRESHOLD:
            return {
                "disease": "Uncertain",
                "confidence": confidence.item(),
                "advice": "Prediction confidence is low. Please try another image."
            }

        # Get advice from DB
        advice = get_advice(normalized_label)

        return {
            "disease": label,
            "confidence": confidence.item(),
            "advice": advice
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {
            "disease": "Error",
            "confidence": 0.0,
            "advice": "There was an error processing your request."
        }

@app.get("/")
async def root():
    return {"message": "Organic Advisor Backend is running!"}
