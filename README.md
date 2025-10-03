# 🌱 Plant Disease Classification (MobileNetV2 + PyTorch)

This project classifies plant leaf diseases using **MobileNetV2** with transfer learning in **PyTorch**.  
It supports **checkpointing**, **resumable training**, and **easy inference**.

---

## 📂 Project Structure
.
├── dataset/
│ └── New Plant Diseases Dataset(Augmented)/
│ ├── train/
│ └── valid/
├── train_model.py
├── best_model/
│ ├── model.pth # Best model
│ └── checkpoint.pth # Last checkpoint
├── labels.json # Class labels
└── README.md

---

## ⚙️ Setup
Install dependencies:
```bash
pip install torch torchvision torchaudio tqdm pillow
python train_model.py
```
Saves:

Best model → best_model/model.pth

Checkpoint → best_model/checkpoint.pth

Labels → labels.json

Resume Training
If training stops, simply run again:

bash
Copy code
python train_model.py

📊 Training Notes

At least 5 epochs recommended for stable accuracy

You can test after 1–2 epochs, but predictions may be less reliable

Checkpoints allow you to stop and resume without losing progress

👨‍💻 Contributors

SaiTej Barla
A. Gowtham
Abhijeet Pastay
B. Praneeth
