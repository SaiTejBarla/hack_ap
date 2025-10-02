import sqlite3
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ===== Step 1: Create database and advisories =====
conn = sqlite3.connect("database.db")  # Fixed path
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS advisories (
    disease_label TEXT PRIMARY KEY,
    advice TEXT
)
""")

advisories = [
    ("Apple_scab", "Use organic fungicide and maintain leaf hygiene."),
    ("Tomato_Bacterial_spot", "Remove infected leaves, use copper-based sprays."),
    ("Healthy", "No disease detected. Keep monitoring your crop.")
]

cursor.executemany(
    "INSERT OR REPLACE INTO advisories (disease_label, advice) VALUES (?, ?)",
    advisories
)

conn.commit()
conn.close()
print("[INFO] Database and advisories table created successfully.")

# ===== Step 2: Extract labels from Hugging Face model =====
print("[INFO] Extracting labels from model...")
model = AutoModelForImageClassification.from_pretrained(
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

labels = model.config.id2label

with open("labels.json", "w") as f:
    json.dump(labels, f, indent=4)

print("[INFO] labels.json created successfully.")
