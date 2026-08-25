from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run detection
results = model.predict(
    source="videos/f4.mp4",
    conf=0.40,
    save=True
)

print("Detection completed.")