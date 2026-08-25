import cv2
from ultralytics import YOLO
import time

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

FIRE_CONFIDENCE = 0.60
REQUIRED_FRAMES = 5

fire_frames = 0
alarm_on = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=FIRE_CONFIDENCE,
        verbose=False
    )

    fire_detected = False

    for box in results[0].boxes:

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        if class_name.lower() == "fire":
            fire_detected = True

    if fire_detected:

        fire_frames += 1

    else:

        fire_frames = 0
        alarm_on = False

    # Confirm fire
    if fire_frames >= REQUIRED_FRAMES:

        if not alarm_on:

            print("🔥🔥 FIRE CONFIRMED! 🔥🔥")

            # Put your alarm code here
            # GPIO / Arduino / ESP32 / MQTT etc.

            alarm_on = True

    annotated_frame = results[0].plot()

    if alarm_on:

        cv2.putText(
            annotated_frame,
            "FIRE ALARM!",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

    cv2.imshow(
        "Fire Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()