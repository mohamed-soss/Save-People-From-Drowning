import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import miniaudio
from mutagen.mp3 import MP3
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple Tracker class if custom tracker is missing
try:
    from tracker import Tracker
except ImportError:
    logging.warning("Custom Tracker not found. Using simple tracker.")
    class Tracker:
        def __init__(self):
            self.next_id = 0
            self.objects = {}
        def update(self, boxes):
            boxes_ids = []
            for box in boxes:
                x1, y1, x2, y2 = box
                boxes_ids.append([x1, y1, x2, y2, self.next_id])
                self.next_id += 1
            return boxes_ids

# Alarm setup
ALARM_PATH = "accident.mp3"
try:
    audio = MP3(ALARM_PATH)
    alarm_length = audio.info.length
except Exception as e:
    logging.error(f"Failed to load alarm file: {e}")
    alarm_length = 5
last_alarm_time = 0
cooldown = 10

def play_alarm():
    try:
        stream = miniaudio.stream_file(ALARM_PATH)
        with miniaudio.PlaybackDevice() as device:
            device.start(stream)
            time.sleep(alarm_length)
    except Exception as e:
        logging.error(f"Alarm playback failed: {e}")

def calc_angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180 / np.pi)
        return 360 - angle if angle > 180 else angle
    except Exception as e:
        logging.error(f"Angle calculation failed: {e}")
        return 0

# Load YOLO model
try:
    model = YOLO("yolov8m.pt")
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    exit(1)

# MediaPipe pose detection
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1)
mpDraw = mp.solutions.drawing_utils
tracker = Tracker()

# Detection setup
drowning_frame_check = 10  # Reduced for faster detection
drowning_flag = 0
swimming_flag = 0
swimming_frame_check = 10
motion_history = {
    'left_wrist': [], 'right_wrist': [], 'left_shoulder': [], 'right_shoulder': [],
    'left_knee': [], 'right_knee': []
}
history_size = 10
drowning_motion_threshold = 0.02
swimming_motion_threshold = 0.03

# Webcam setup
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        logging.info(f"Webcam opened on index {i}.")
        break
if not cap or not cap.isOpened():
    logging.error("Failed to open webcam.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read frame from webcam.")
        break

    h, w, _ = frame.shape
    try:
        results = model(frame)[0]
    except Exception as e:
        logging.error(f"YOLO inference failed: {e}")
        continue

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(imgRGB)

    if pose_result.pose_landmarks:
        mpDraw.draw_landmarks(frame, pose_result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        landmarks = pose_result.pose_landmarks.landmark
        logging.debug("Pose landmarks detected.")

        def get_xy(p):
            try:
                return [landmarks[p].x, landmarks[p].y]
            except Exception as e:
                logging.error(f"Failed to get landmark {p}: {e}")
                return [0, 0]

        l_shoulder = get_xy(mpPose.PoseLandmark.LEFT_SHOULDER.value)
        l_elbow = get_xy(mpPose.PoseLandmark.LEFT_ELBOW.value)
        l_wrist = get_xy(mpPose.PoseLandmark.LEFT_WRIST.value)
        r_shoulder = get_xy(mpPose.PoseLandmark.RIGHT_SHOULDER.value)
        r_elbow = get_xy(mpPose.PoseLandmark.RIGHT_ELBOW.value)
        r_wrist = get_xy(mpPose.PoseLandmark.RIGHT_WRIST.value)
        l_knee = get_xy(mpPose.PoseLandmark.LEFT_KNEE.value)
        r_knee = get_xy(mpPose.PoseLandmark.RIGHT_KNEE.value)

        # Arm angles
        l_arm_angle = calc_angle(l_shoulder, l_elbow, l_wrist)
        r_arm_angle = calc_angle(r_shoulder, r_elbow, r_wrist)
        cv2.putText(frame, f'L Arm: {int(l_arm_angle)}', tuple(np.multiply(l_elbow, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, f'R Arm: {int(r_arm_angle)}', tuple(np.multiply(r_elbow, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        # Update motion history
        keypoints = {
            'left_wrist': l_wrist, 'right_wrist': r_wrist, 'left_shoulder': l_shoulder,
            'right_shoulder': r_shoulder, 'left_knee': l_knee, 'right_knee': r_knee
        }
        for key in motion_history:
            motion_history[key].append(keypoints[key])
            if len(motion_history[key]) > history_size:
                motion_history[key].pop(0)

        # Calculate average speeds
        avg_speed = {}
        for key, points in motion_history.items():
            if len(points) >= 2:
                speeds = [np.linalg.norm(np.array(points[i]) - np.array(points[i - 1]))
                          for i in range(1, len(points))]
                avg_speed[key] = sum(speeds) / len(speeds) if speeds else 0
                cv2.putText(frame, f'{key}: {avg_speed[key]:.3f}', (20, 150 + 20 * list(motion_history.keys()).index(key)),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

        # Overall body motion
        body_motion = sum(avg_speed.values()) / len(avg_speed) if avg_speed else 0
        cv2.putText(frame, f'Body Motion: {body_motion:.3f}', (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

        # Swimming detection
        arm_motion = (avg_speed.get('left_wrist', 0) + avg_speed.get('right_wrist', 0)) / 2
        leg_motion = (avg_speed.get('left_knee', 0) + avg_speed.get('right_knee', 0)) / 2
        cv2.putText(frame, f'Leg Motion: {leg_motion:.3f}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        if arm_motion > swimming_motion_threshold and leg_motion > swimming_motion_threshold:
            swimming_flag += 1
            if swimming_flag >= swimming_frame_check:
                cv2.putText(frame, 'SWIMMING DETECTED', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                drowning_flag = 0
                logging.debug("Swimming detected.")
        else:
            swimming_flag = 0

        # Drowning detection
        l_arm_raised = l_wrist[1] < l_elbow[1] < l_shoulder[1] and l_arm_angle > 100
        r_arm_raised = r_wrist[1] < r_elbow[1] < r_shoulder[1] and r_arm_angle > 100
        minimal_body_motion = body_motion < drowning_motion_threshold
        logging.debug(f"L Arm Raised: {l_arm_raised}, R Arm Raised: {r_arm_raised}, "
                      f"Body Motion: {body_motion:.3f}, Minimal: {minimal_body_motion}")
        if (l_arm_raised or r_arm_raised) and minimal_body_motion and swimming_flag == 0:
            drowning_flag += 1
            cv2.putText(frame, f'Drowning Flag: {drowning_flag}/{drowning_frame_check}', (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            if drowning_flag >= drowning_frame_check and time.time() - last_alarm_time > cooldown:
                cv2.putText(frame, 'DROWNING DETECTED!', (20, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                threading.Thread(target=play_alarm).start()
                last_alarm_time = time.time()
                logging.info("Drowning detected!")
        else:
            drowning_flag = 0
            logging.debug(f"Drowning not detected. Flag: {drowning_flag}")

    # Person detection
    points = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf > 0.2:
            points.append([int(x1), int(y1), int(x2), int(y2)])

    # Track and draw
    boxes_ids = tracker.update(points)
    for x, y, x2, y2, pid in boxes_ids:
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    # Display number of people
    cv2.putText(frame, f'Persons: {len(points)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    cv2.imshow("Live Drowning Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()