from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Detect fire
results = model.predict(
    source="test.jpg",
    conf=0.40,
    save=True
)

print("Detection completed.")