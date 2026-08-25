from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run detection
results = model.predict(
    source="images/1.jpg",
    conf=0.40,
    save=True
)

print("Detection completed.")