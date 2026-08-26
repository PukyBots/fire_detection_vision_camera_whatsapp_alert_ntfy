import cv2
import os
import time
import threading
from ultralytics import YOLO


# ============================================================
# OPENCV / FFMPEG RTSP SETTINGS
# ============================================================

os.environ[
    "OPENCV_FFMPEG_CAPTURE_OPTIONS"
] = "rtsp_transport;tcp|stimeout;5000000"


# ============================================================
# RTSP CAMERAS
# ============================================================

RTSP_URLS = [

    "rtsp://pulkitgarg:Allenhouse@123@192.168.1.57:554/stream1",

    "rtsp://pulkitgarg:Allenhouse@123@192.168.1.60:554/stream2",

    # Add more cameras here
]


# ============================================================
# CAMERA SHARED DATA
# ============================================================

camera_frames = [None] * len(RTSP_URLS)

camera_online = [False] * len(RTSP_URLS)

camera_last_frame_time = [0] * len(RTSP_URLS)

frame_locks = [
    threading.Lock()
    for _ in RTSP_URLS
]


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(
    "runs/detect/train/weights/best.pt"
)


# ============================================================
# RTSP CAMERA READER
# ============================================================

def camera_reader(cam_index):

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

            # Optional buffer reduction
            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            if not cap.isOpened():

                print(
                    f"[CAM {cam_index + 1}] Offline"
                )

                camera_online[cam_index] = False

                time.sleep(5)

                continue


            print(
                f"[CAM {cam_index + 1}] Connected"
            )

            camera_online[cam_index] = True


            # ====================================================
            # READ FRAMES
            # ====================================================

            while True:

                ret, frame = cap.read()

                if not ret:

                    print(
                        f"[CAM {cam_index + 1}] Lost connection"
                    )

                    camera_online[cam_index] = False

                    with frame_locks[cam_index]:

                        camera_frames[cam_index] = None

                    break


                # Store ONLY latest frame
                with frame_locks[cam_index]:

                    camera_frames[cam_index] = frame.copy()


                camera_last_frame_time[cam_index] = time.time()

                camera_online[cam_index] = True


        except Exception as e:

            print(
                f"[CAM {cam_index + 1}] Error: {e}"
            )


        # ========================================================
        # CLEANUP AFTER DISCONNECT
        # ========================================================

        camera_online[cam_index] = False

        with frame_locks[cam_index]:

            camera_frames[cam_index] = None


        try:

            if cap is not None:
                cap.release()

        except Exception:
            pass


        cap = None


        print(
            f"[CAM {cam_index + 1}] Reconnecting in 5 seconds..."
        )

        time.sleep(5)


# ============================================================
# START CAMERA THREADS
# ============================================================

camera_threads = []


for cam_index in range(len(RTSP_URLS)):

    thread = threading.Thread(
        target=camera_reader,
        args=(cam_index,),
        daemon=True
    )

    thread.start()

    camera_threads.append(thread)


# ============================================================
# MAIN YOLO PROCESSING LOOP
# ============================================================

try:

    while True:

        for cam_index in range(len(RTSP_URLS)):

            # ------------------------------------------------
            # Get latest frame
            # ------------------------------------------------

            with frame_locks[cam_index]:

                if camera_frames[cam_index] is None:

                    frame = None

                else:

                    frame = camera_frames[
                        cam_index
                    ].copy()


            # ------------------------------------------------
            # Camera offline / no frame
            # ------------------------------------------------

            if frame is None:

                continue


            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------

            results = model.predict(

                source=frame,

                conf=0.40,

                verbose=False

            )


            # ------------------------------------------------
            # Draw detections
            # ------------------------------------------------

            annotated_frame = results[0].plot()


            # ------------------------------------------------
            # Add camera information
            # ------------------------------------------------

            cv2.putText(

                annotated_frame,

                f"CAM {cam_index + 1}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2

            )


            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(

                f"Camera {cam_index + 1}",

                annotated_frame

            )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


finally:

    cv2.destroyAllWindows()

    print("Program stopped.")