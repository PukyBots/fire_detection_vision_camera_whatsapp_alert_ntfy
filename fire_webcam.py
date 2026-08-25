import cv2
from ultralytics import YOLO

# Load trained fire model
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    # Run YOLO detection
    results = model.predict(
        source=frame,
        conf=0.40,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Fire Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()