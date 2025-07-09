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

def draw_label(img, text, pos, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 6), (x + text_w + 6, y), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, font_scale, color, thickness, cv2.LINE_AA)

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
drowning_frame_check = 5
drowning_flag = 0
swimming_flag = 0
swimming_frame_check = 5
motion_history = {
    'left_wrist': [], 'right_wrist': [], 'left_shoulder': [], 'right_shoulder': [],
    'left_knee': [], 'right_knee': []
}
history_size = 6
drowning_motion_threshold = 0.02
swimming_motion_threshold = 0.03
motion_flag = 0
motion_frame_check = 5
motion_threshold = 0.02

# Load video file
cap = cv2.VideoCapture("1.mp4")
if not cap.isOpened():
    logging.error("Failed to open video file.")
    exit(1)

cv2.namedWindow("Drowning Detection (Video)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drowning Detection (Video)", 700, 700)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.info("End of video reached.")
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

        l_ang = calc_angle(l_shoulder, l_elbow, l_wrist)
        r_ang = calc_angle(r_shoulder, r_elbow, r_wrist)

        keypoints = {
            'left_wrist': l_wrist, 'right_wrist': r_wrist, 'left_shoulder': l_shoulder,
            'right_shoulder': r_shoulder, 'left_knee': l_knee, 'right_knee': r_knee
        }
        for key in motion_history:
            motion_history[key].append(keypoints[key])
            if len(motion_history[key]) > history_size:
                motion_history[key].pop(0)

        avg_speed = {}
        for key, points in motion_history.items():
            if len(points) >= 2:
                speeds = [np.linalg.norm(np.array(points[i]) - np.array(points[i - 1]))
                          for i in range(1, len(points))]
                avg_speed[key] = sum(speeds) / len(speeds) if speeds else 0

        body_motion = sum(avg_speed.values()) / len(avg_speed) if avg_speed else 0
        arm_motion = (avg_speed.get('left_wrist', 0) + avg_speed.get('right_wrist', 0)) / 2
        leg_motion = (avg_speed.get('left_knee', 0) + avg_speed.get('right_knee', 0)) / 2

        if arm_motion > swimming_motion_threshold and leg_motion > swimming_motion_threshold:
            swimming_flag += 1
            if swimming_flag >= swimming_frame_check:
                draw_label(frame, '✅ Swimming Detected', (20, 75), 1.0, (0, 255, 0), (0, 50, 0))
                drowning_flag = 0
        else:
            swimming_flag = 0

        l_condition = l_wrist[1] < l_elbow[1] < l_shoulder[1] and l_ang > 100
        r_condition = r_wrist[1] < r_elbow[1] < r_shoulder[1] and r_ang > 100
        minimal_body_motion = body_motion < drowning_motion_threshold

        if (l_condition or r_condition) and minimal_body_motion and swimming_flag == 0:
            drowning_flag += 1
            draw_label(frame, f'Drowning Check: {drowning_flag}/{drowning_frame_check}', (20, 100), 0.8, (255, 255, 255), (0, 0, 128))
            if drowning_flag >= drowning_frame_check and time.time() - last_alarm_time > cooldown:
                draw_label(frame, '🚨 DROWNING DETECTED', (20, 140), 1.1, (0, 0, 255), (0, 0, 50))
                threading.Thread(target=play_alarm).start()
                last_alarm_time = time.time()
        else:
            drowning_flag = 0

        fast_keys = [k for k, v in avg_speed.items() if v > motion_threshold]
        if len(fast_keys) >= 2:
            motion_flag += 1
            if motion_flag >= motion_frame_check and time.time() - last_alarm_time > cooldown:
                draw_label(frame, '⚠️ Fast Random Movement!', (20, 180), 1.0, (0, 0, 255), (50, 0, 0))
                threading.Thread(target=play_alarm).start()
                last_alarm_time = time.time()
        else:
            motion_flag = 0

    points = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf > 0.2:
            points.append([int(x1), int(y1), int(x2), int(y2)])

    boxes_ids = tracker.update(points)
    for x, y, x2, y2, pid in boxes_ids:
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    draw_label(frame, f'Persons Detected: {len(points)}', (20, 40), 0.8, (255, 255, 255), (128, 0, 128))

    frame_resized = cv2.resize(frame, (500, 500))
    cv2.imshow("Drowning Detection (Video)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
