from transformers import pipeline
import os

pipe = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

def identify_disease(image_path):
    if not os.path.exists(image_path):
        return {"error": "Image not found."}

    try:
        results = pipe(image_path)
        return results
    except Exception as e:
        return {"error": str(e)}
