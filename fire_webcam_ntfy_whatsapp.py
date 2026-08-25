import cv2
import requests
import time
import threading

from ultralytics import YOLO

import cloudinary
import cloudinary.uploader

from twilio.rest import Client
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

CAMERA_INDEX = 0

CONFIDENCE = 0.40

# Number of consecutive frames required
# before confirming a fire
fire_CONFIRMATION_FRAMES = 5

# Prevent repeated alerts
ALERT_COOLDOWN = 5


# ============================================================
# NTFY CONFIGURATION
# ============================================================

NTFY_TOPIC = "pulkit-fire-detected"

NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


# ============================================================
# CLOUDINARY CONFIGURATION
# ============================================================

cloudinary.config(
    cloud_name="dl2fhwcl5",
    api_key="971978761231223",
    api_secret="G71zlwDG-zH65inED7kJn55px1M"
)


# ============================================================
# TWILIO
# ============================================================

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

twilio_client = Client(
    ACCOUNT_SID,
    AUTH_TOKEN
)

# Twilio WhatsApp Sandbox number
TWILIO_WHATSAPP = "whatsapp:+14155238886"


# Caregiver WhatsApp numbers
CAREGIVERS = [
    "whatsapp:+918953193403",
]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

fire_counter = 0

fire_alert_sent = False

last_alert_time = 0

alert_in_progress = False


# ============================================================
# NTFY FUNCTION
# ============================================================

def send_ntfy():

    print()
    print("========== NTFY START ==========")

    try:

        response = requests.post(
            NTFY_URL,
            data=(
                "⚠ fire DETECTED\n\n"
                "fire detected by Camera 1.\n"
            ).encode("utf-8"),

            headers={
                "Title": "fire DETECTED",
                "Priority": "urgent",
                "Tags": "warning"
            },

            timeout=10
        )

        print("[NTFY] HTTP STATUS:", response.status_code)
        print("[NTFY] RESPONSE:", response.text)

        if response.status_code == 200:
            print("[NTFY] >>> SUCCESS")
        else:
            print("[NTFY] >>> FAILED")

    except requests.exceptions.RequestException as e:

        print("[NTFY] REQUEST ERROR:")
        print(repr(e))

    except Exception as e:

        print("[NTFY] UNKNOWN ERROR:")
        print(repr(e))

    print("========== NTFY END ==========")


# ============================================================
# CLOUDINARY UPLOAD
# ============================================================

def upload_frame_to_cloudinary(frame):

    print()
    print(
        "[CLOUDINARY] Uploading fire image..."
    )

    try:

        # Convert OpenCV frame to JPG
        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not success:

            print(
                "[CLOUDINARY] Could not encode image"
            )

            return None


        # Upload to Cloudinary
        result = cloudinary.uploader.upload(

            buffer.tobytes(),

            resource_type="image",

            format="jpg",

            quality="auto:good"
        )


        image_url = result["secure_url"]


        print(
            "[CLOUDINARY] Upload successful"
        )

        print(
            "[CLOUDINARY] URL:",
            image_url
        )


        return image_url


    except Exception as e:

        print(
            "[CLOUDINARY] Upload failed:",
            e
        )

        return None


# ============================================================
# WHATSAPP FUNCTION
# ============================================================

def send_whatsapp_alert(image_url):

    print()
    print("========== WHATSAPP START ==========")

    print("[WHATSAPP] Image URL:")
    print(image_url)

    try:

        for number in CAREGIVERS:

            print(
                f"[WHATSAPP] Sending to {number}"
            )

            message = twilio_client.messages.create(

                from_=TWILIO_WHATSAPP,

                to=number,

                body=(
                    "⚠ fire DETECTED\n\n"
                    "Camera: 1\n"
                    f"Time: {time.strftime('%H:%M:%S')}\n\n"
                ),

                media_url=[image_url]
            )

            print(
                "[WHATSAPP] SUCCESS"
            )

            print(
                "[WHATSAPP] SID:",
                message.sid
            )

            print(
                "[WHATSAPP] STATUS:",
                message.status
            )

    except Exception as e:

        print(
            "[WHATSAPP] ERROR:"
        )

        print(
            repr(e)
        )

    print("========== WHATSAPP END ==========")

# ============================================================
# COMPLETE ALERT PROCESS
# ============================================================

def process_alert(frame):

    global alert_in_progress

    print()
    print("==========================================")
    print("       PROCESSING fire ALERT")
    print("==========================================")

    try:

        # ==================================================
        # 1. NTFY
        # ==================================================

        print()
        print("[1/3] Sending NTFY...")

        send_ntfy()


        # ==================================================
        # 2. CLOUDINARY
        # ==================================================

        print()
        print("[2/3] Uploading image to Cloudinary...")

        image_url = upload_frame_to_cloudinary(frame)


        # ==================================================
        # 3. WHATSAPP
        # ==================================================

        if image_url:

            print()
            print("[3/3] Sending WhatsApp...")

            send_whatsapp_alert(
                image_url
            )

        else:

            print(
                "[3/3] WhatsApp skipped - "
                "no Cloudinary URL"
            )


    except Exception as e:

        print()
        print("[ALERT PROCESS ERROR]")
        print(repr(e))


    finally:

        alert_in_progress = False

        print()
        print("==========================================")
        print("       ALERT PROCESS FINISHED")
        print("==========================================")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("[SYSTEM] Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("[SYSTEM] Model loaded successfully")
print("[SYSTEM] Classes:", model.names)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print(
        "[ERROR] Could not open webcam"
    )

    exit()


print(
    "[SYSTEM] Camera started."
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    ret, frame = cap.read()


    if not ret:

        print(
            "[ERROR] Could not read camera frame"
        )

        break


    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    results = model.predict(

        frame,

        conf=CONFIDENCE,

        verbose=False
    )


    # --------------------------------------------------------
    # Draw YOLO detections
    # --------------------------------------------------------

    annotated_frame = results[0].plot()


    # --------------------------------------------------------
    # Check for fire
    # --------------------------------------------------------

    detected_fire = False


    for box in results[0].boxes:


        class_id = int(
            box.cls[0]
        )


        confidence = float(
            box.conf[0]
        )


        class_name = model.names[
            class_id
        ]


        print(
            f"YOLO: {class_name} | "
            f"confidence={confidence:.2f}"
        )


        # Your YAML class name is:
        #
        # fire-Detected

        if class_name.lower() == "fire":

            detected_fire = True


    # --------------------------------------------------------
    # fire CONFIRMATION
    # --------------------------------------------------------

    if detected_fire:

        fire_counter += 1


        print(
            f"[fire] Confirmation "
            f"{fire_counter}/"
            f"{fire_CONFIRMATION_FRAMES}"
        )


    else:

        if fire_counter > 0:

            print(
                "[fire] Detection lost - resetting"
            )


        fire_counter = 0


    # --------------------------------------------------------
    # CONFIRMED fire
    # --------------------------------------------------------

    if (

        fire_counter >= fire_CONFIRMATION_FRAMES

        and not fire_alert_sent

        and not alert_in_progress

        and (
            time.time() - last_alert_time
            > ALERT_COOLDOWN
        )

    ):


        print()
        print(
            "========================================"
        )

        print(
            "             fire CONFIRMED"
        )

        print(
            "========================================"
        )


        # Take a copy of the current frame
        alert_frame = frame.copy()


        # Prevent multiple alerts
        alert_in_progress = True

        fire_alert_sent = True


        # Record alert time
        last_alert_time = time.time()


        # ----------------------------------------------------
        # Run alert processing in background
        # ----------------------------------------------------

        threading.Thread(

            target=process_alert,

            args=(alert_frame,),

            daemon=True

        ).start()


        # Reset counter
        fire_counter = 0


    # --------------------------------------------------------
    # Reset after fire disappears
    # --------------------------------------------------------

    if not detected_fire:

        fire_alert_sent = False


    # --------------------------------------------------------
    # Display camera
    # --------------------------------------------------------

    cv2.imshow(

        "fire Detection",

        annotated_frame

    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print(
    "[SYSTEM] Program stopped."
)