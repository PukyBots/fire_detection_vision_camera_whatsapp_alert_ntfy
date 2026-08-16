from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Train on your Roboflow fire dataset
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0
)

print("Training completed!")