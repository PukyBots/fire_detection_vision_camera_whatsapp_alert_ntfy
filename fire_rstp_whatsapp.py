import cv2
import requests
import time
import threading
import os

from ultralytics import YOLO

import cloudinary
import cloudinary.uploader

from twilio.rest import Client


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

CONFIDENCE = 0.40

# Number of consecutive frames required
# before confirming fire
FIRE_CONFIRMATION_FRAMES = 5

# Prevent repeated alerts
ALERT_COOLDOWN = 5


# ============================================================
# RTSP CONFIGURATION
# ============================================================

os.environ[
    "OPENCV_FFMPEG_CAPTURE_OPTIONS"
] = "rtsp_transport;tcp|stimeout;5000000"


RTSP_URLS = [

    "rtsp://pulkitgarg:Allenhouse@123@192.168.1.57:554/stream1",

    "rtsp://pulkitgarg:Allenhouse@123@192.168.1.60:554/stream2",

    # Add more cameras here

]


# ============================================================
# CAMERA SHARED DATA
# ============================================================

camera_frames = [
    None
    for _ in RTSP_URLS
]

camera_online = [
    False
    for _ in RTSP_URLS
]

camera_last_frame_time = [
    0
    for _ in RTSP_URLS
]

frame_locks = [
    threading.Lock()
    for _ in RTSP_URLS
]


# ============================================================
# FIRE DETECTION STATE
# ============================================================

fire_counters = [
    0
    for _ in RTSP_URLS
]

fire_alert_sent = [
    False
    for _ in RTSP_URLS
]

last_alert_times = [
    0
    for _ in RTSP_URLS
]

alert_in_progress = [
    False
    for _ in RTSP_URLS
]


# ============================================================
# NTFY CONFIGURATION
# ============================================================

NTFY_TOPIC = "pulkit-fire-detected"

NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


# ============================================================
# CLOUDINARY CONFIGURATION
# ============================================================

cloudinary.config(

    cloud_name="YOUR_CLOUD_NAME",

    api_key="YOUR_API_KEY",

    api_secret="YOUR_API_SECRET"
)


# ============================================================
# TWILIO CONFIGURATION
# ============================================================

ACCOUNT_SID = os.getenv(
    "TWILIO_ACCOUNT_SID"
)

AUTH_TOKEN = os.getenv(
    "TWILIO_AUTH_TOKEN"
)


twilio_client = Client(
    ACCOUNT_SID,
    AUTH_TOKEN
)


# Twilio WhatsApp Sandbox
TWILIO_WHATSAPP = "whatsapp:+14155238886"


# Caregiver WhatsApp numbers
CAREGIVERS = [

    "whatsapp:+918953193403",

]


# ============================================================
# NTFY FUNCTION
# ============================================================

def send_ntfy(cam_index):

    print()
    print("========== NTFY START ==========")

    try:

        camera_number = cam_index + 1

        response = requests.post(

            NTFY_URL,

            data=(
                "⚠ FIRE DETECTED\n\n"
                f"Fire detected by Camera {camera_number}.\n"
                f"Time: {time.strftime('%H:%M:%S')}\n"
            ).encode("utf-8"),

            headers={

                "Title": (
                    f"🔥 FIRE DETECTED - "
                    f"CAMERA {camera_number}"
                ),

                "Priority": "urgent",

                "Tags": "warning,fire"

            },

            timeout=10
        )


        print(
            "[NTFY] HTTP STATUS:",
            response.status_code
        )

        print(
            "[NTFY] RESPONSE:",
            response.text
        )


        if response.status_code == 200:

            print(
                "[NTFY] >>> SUCCESS"
            )

        else:

            print(
                "[NTFY] >>> FAILED"
            )


    except requests.exceptions.RequestException as e:

        print(
            "[NTFY] REQUEST ERROR:"
        )

        print(
            repr(e)
        )


    except Exception as e:

        print(
            "[NTFY] UNKNOWN ERROR:"
        )

        print(
            repr(e)
        )


    print(
        "========== NTFY END =========="
    )


# ============================================================
# CLOUDINARY UPLOAD
# ============================================================

def upload_frame_to_cloudinary(
    frame,
    cam_index
):

    print()

    print(
        f"[CLOUDINARY] Uploading "
        f"Camera {cam_index + 1} image..."
    )


    try:

        # ----------------------------------------------------
        # Convert OpenCV frame to JPG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not success:

            print(
                "[CLOUDINARY] Could not encode image"
            )

            return None


        # ----------------------------------------------------
        # Upload
        # ----------------------------------------------------

        result = cloudinary.uploader.upload(

            buffer.tobytes(),

            resource_type="image",

            format="jpg",

            quality="auto:good"
        )


        image_url = result[
            "secure_url"
        ]


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

def send_whatsapp_alert(
    image_url,
    cam_index
):

    camera_number = cam_index + 1


    print()

    print(
        "========== WHATSAPP START =========="
    )


    print(
        "[WHATSAPP] Image URL:"
    )

    print(
        image_url
    )


    try:

        for number in CAREGIVERS:

            print(
                f"[WHATSAPP] Sending to {number}"
            )


            message = twilio_client.messages.create(

                from_=TWILIO_WHATSAPP,

                to=number,

                body=(

                    "⚠ FIRE DETECTED\n\n"

                    f"Camera: {camera_number}\n"

                    f"Time: "
                    f"{time.strftime('%H:%M:%S')}\n"

                ),

                media_url=[
                    image_url
                ]

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


    print(
        "========== WHATSAPP END =========="
    )


# ============================================================
# COMPLETE ALERT PROCESS
# ============================================================

def process_alert(
    frame,
    cam_index
):

    print()

    print(
        "=========================================="
    )

    print(
        f"PROCESSING FIRE ALERT - "
        f"CAMERA {cam_index + 1}"
    )

    print(
        "=========================================="
    )


    try:

        # ====================================================
        # 1. NTFY
        # ====================================================

        print()

        print(
            "[1/3] Sending NTFY..."
        )


        send_ntfy(
            cam_index
        )


        # ====================================================
        # 2. CLOUDINARY
        # ====================================================

        print()

        print(
            "[2/3] Uploading image..."
        )


        image_url = upload_frame_to_cloudinary(

            frame,

            cam_index

        )


        # ====================================================
        # 3. WHATSAPP
        # ====================================================

        if image_url:

            print()

            print(
                "[3/3] Sending WhatsApp..."
            )


            send_whatsapp_alert(

                image_url,

                cam_index

            )


        else:

            print(
                "[3/3] WhatsApp skipped - "
                "no Cloudinary URL"
            )


    except Exception as e:

        print()

        print(
            "[ALERT PROCESS ERROR]"
        )

        print(
            repr(e)
        )


    finally:

        alert_in_progress[
            cam_index
        ] = False


        print()

        print(
            "=========================================="
        )

        print(
            f"ALERT PROCESS FINISHED - "
            f"CAMERA {cam_index + 1}"
        )

        print(
            "=========================================="
        )


# ============================================================
# RTSP CAMERA READER
# ============================================================

def camera_reader(
    cam_index
):

    cap = None


    while True:

        try:

            print(
                f"[CAM {cam_index + 1}] Connecting..."
            )


            cap = cv2.VideoCapture(

                RTSP_URLS[cam_index],

                cv2.CAP_FFMPEG

            )


            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )


            if not cap.isOpened():

                print(
                    f"[CAM {cam_index + 1}] Offline"
                )


                camera_online[
                    cam_index
                ] = False


                time.sleep(5)

                continue


            print(
                f"[CAM {cam_index + 1}] Connected"
            )


            camera_online[
                cam_index
            ] = True


            # =================================================
            # READ RTSP FRAMES
            # =================================================

            while True:

                ret, frame = cap.read()


                if not ret:

                    print(
                        f"[CAM {cam_index + 1}] "
                        f"Lost connection"
                    )


                    camera_online[
                        cam_index
                    ] = False


                    with frame_locks[
                        cam_index
                    ]:

                        camera_frames[
                            cam_index
                        ] = None


                    break


                # ------------------------------------------------
                # Store latest frame only
                # ------------------------------------------------

                with frame_locks[
                    cam_index
                ]:

                    camera_frames[
                        cam_index
                    ] = frame.copy()


                camera_last_frame_time[
                    cam_index
                ] = time.time()


                camera_online[
                    cam_index
                ] = True


        except Exception as e:

            print(
                f"[CAM {cam_index + 1}] "
                f"ERROR: {e}"
            )


        # ========================================================
        # CLEANUP
        # ========================================================

        camera_online[
            cam_index
        ] = False


        with frame_locks[
            cam_index
        ]:

            camera_frames[
                cam_index
            ] = None


        try:

            if cap is not None:

                cap.release()

        except Exception:

            pass


        cap = None


        print(
            f"[CAM {cam_index + 1}] "
            f"Reconnecting in 5 seconds..."
        )


        time.sleep(5)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print(
    "[SYSTEM] Loading YOLO model..."
)


model = YOLO(
    MODEL_PATH
)


print(
    "[SYSTEM] Model loaded successfully"
)


print(
    "[SYSTEM] Classes:",
    model.names
)


# ============================================================
# START RTSP CAMERA THREADS
# ============================================================

print()

print(
    "[SYSTEM] Starting RTSP cameras..."
)


camera_threads = []


for cam_index in range(
    len(RTSP_URLS)
):

    thread = threading.Thread(

        target=camera_reader,

        args=(cam_index,),

        daemon=True

    )


    thread.start()


    camera_threads.append(
        thread
    )


# ============================================================
# MAIN YOLO PROCESSING LOOP
# ============================================================

try:

    while True:

        for cam_index in range(
            len(RTSP_URLS)
        ):

            # =================================================
            # GET LATEST FRAME
            # =================================================

            with frame_locks[
                cam_index
            ]:

                if (
                    camera_frames[
                        cam_index
                    ]
                    is None
                ):

                    frame = None

                else:

                    frame = camera_frames[
                        cam_index
                    ].copy()


            # =================================================
            # CAMERA OFFLINE
            # =================================================

            if frame is None:

                continue


            # =================================================
            # YOLO PREDICTION
            # =================================================

            results = model.predict(

                source=frame,

                conf=CONFIDENCE,

                verbose=False

            )


            # =================================================
            # DRAW DETECTIONS
            # =================================================

            annotated_frame = (
                results[0].plot()
            )


            # =================================================
            # CAMERA LABEL
            # =================================================

            status_text = (

                f"CAM {cam_index + 1} | "

                + (
                    "ONLINE"
                    if camera_online[
                        cam_index
                    ]
                    else "OFFLINE"
                )

            )


            cv2.putText(

                annotated_frame,

                status_text,

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2

            )


            # =================================================
            # CHECK FOR FIRE
            # =================================================

            detected_fire = False


            for box in results[
                0
            ].boxes:


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

                    f"CAM {cam_index + 1}: "

                    f"{class_name} | "

                    f"confidence="
                    f"{confidence:.2f}"

                )


                # ------------------------------------------------
                # Fire class
                # ------------------------------------------------

                if (
                    class_name.lower()
                    == "fire"
                ):

                    detected_fire = True


            # =================================================
            # FIRE CONFIRMATION
            # =================================================

            if detected_fire:

                fire_counters[
                    cam_index
                ] += 1


                print(

                    f"[CAM {cam_index + 1}] "

                    f"Fire confirmation "

                    f"{fire_counters[cam_index]}/"

                    f"{FIRE_CONFIRMATION_FRAMES}"

                )


            else:

                if (
                    fire_counters[
                        cam_index
                    ] > 0
                ):

                    print(

                        f"[CAM {cam_index + 1}] "

                        "Fire detection lost - "
                        "resetting"

                    )


                fire_counters[
                    cam_index
                ] = 0


            # =================================================
            # CONFIRMED FIRE
            # =================================================

            if (

                fire_counters[
                    cam_index
                ]
                >= FIRE_CONFIRMATION_FRAMES

                and not fire_alert_sent[
                    cam_index
                ]

                and not alert_in_progress[
                    cam_index
                ]

                and (

                    time.time()
                    - last_alert_times[
                        cam_index
                    ]

                    > ALERT_COOLDOWN

                )

            ):


                print()

                print(
                    "========================================"
                )

                print(

                    f"🔥 FIRE CONFIRMED - "
                    f"CAMERA {cam_index + 1}"

                )

                print(
                    "========================================"
                )


                # ------------------------------------------------
                # IMPORTANT:
                # Copy frame BEFORE starting alert thread
                # ------------------------------------------------

                alert_frame = frame.copy()


                # ------------------------------------------------
                # Prevent duplicate alerts
                # ------------------------------------------------

                alert_in_progress[
                    cam_index
                ] = True


                fire_alert_sent[
                    cam_index
                ] = True


                last_alert_times[
                    cam_index
                ] = time.time()


                # ------------------------------------------------
                # Send alert in background
                # ------------------------------------------------

                threading.Thread(

                    target=process_alert,

                    args=(

                        alert_frame,

                        cam_index

                    ),

                    daemon=True

                ).start()


                # Reset confirmation counter
                fire_counters[
                    cam_index
                ] = 0


            # =================================================
            # FIRE DISAPPEARED
            # =================================================

            if not detected_fire:

                fire_alert_sent[
                    cam_index
                ] = False


            # =================================================
            # DISPLAY CAMERA
            # =================================================

            cv2.imshow(

                f"Camera {cam_index + 1}",

                annotated_frame

            )


        # =====================================================
        # QUIT
        # =====================================================

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


finally:

    cv2.destroyAllWindows()


    print(
        "[SYSTEM] Program stopped."
    )