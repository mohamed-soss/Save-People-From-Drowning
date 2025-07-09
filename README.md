# Drowning Detection System

This project implements a real-time drowning detection system using computer vision techniques. It analyzes video footage to detect potential drowning incidents by tracking human poses and movements, utilizing the YOLOv8 model for person detection and MediaPipe for pose estimation. The system triggers an alarm when drowning or fast random movements are detected, making it suitable for monitoring swimming pools or other water bodies.

---

## Features
- **Person Detection**: Uses YOLOv8 to detect people in video frames.
- **Pose Estimation**: Employs MediaPipe to track key body landmarks (e.g., shoulders, elbows, wrists, knees).
- **Drowning Detection**: Identifies drowning behavior based on arm angles and minimal body motion.
- **Swimming Detection**: Recognizes swimming patterns through arm and leg motion analysis.
- **Fast Movement Detection**: Detects rapid, random movements that may indicate distress.
- **Alarm System**: Plays an audio alarm (accident.mp3) when potential drowning or fast movements are detected, with a cooldown period to prevent repetitive alerts.
- **Real-Time Visualization**: Displays annotated video with detection labels and bounding boxes.

---

## Dataset / Input
The system processes a video file (`1.mp4`) for analysis. Make sure this video file is available in the project directory or modify the `cap = cv2.VideoCapture("1.mp4")` line in the code to point to your own video source.

---

## Prerequisites
Install the following Python libraries:

- opencv-python
- numpy
- mediapipe
- miniaudio
- mutagen
- ultralytics
- logging

You can install them with:

```bash
pip install opencv-python numpy mediapipe miniaudio mutagen ultralytics
````

Additionally, download the YOLOv8 model weights (e.g., `yolov8m.pt`) from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics) and place them in your project directory.

---

## Project Structure

* `drowning_detection.py`: Main Python script for the drowning detection system.
* `accident.mp3`: Audio file for the alarm (ensure this file is present).
* `yolov8m.pt`: YOLOv8 model weights for person detection.
* `1.mp4`: Input video file for testing the system.
* `README.md`: This file with an overview of the project.
* `tracker.py` (optional): Custom tracker for object tracking (a simple fallback tracker is included in the main script if not provided).

---

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <project-directory>
   ```

3. Ensure required files (`1.mp4`, `accident.mp3`, `yolov8m.pt`) are present.

4. Run the script:

   ```bash
   python drowning_detection.py
   ```

The script will process the video, display annotated output in a window, and play an alarm if drowning or fast movements are detected. Press `q` to exit the video window.

---

## Technical Details

* **YOLOv8**: Used for detecting persons in video frames with a confidence threshold of 0.2.
* **MediaPipe Pose**: Tracks 33 body landmarks to calculate arm angles and body motion.
* **Motion Analysis**: Computes average speed of key body parts (wrists, shoulders, knees) over 6 frames to detect swimming or drowning.

**Drowning Criteria:**

* Arm angles > 100° with wrists above elbows and shoulders.
* Minimal body motion (below 0.02 threshold).
* No swimming detected (arm/leg motion below 0.03 threshold).

**Swimming Criteria:**

* Arm and leg motion above 0.03 threshold.

**Fast Movement Detection:**

* Two or more body parts moving faster than a 0.02 threshold.

**Alarm Cooldown:**

* 10 seconds to prevent repeated alarms.

**Logging:**

* Uses Python's logging module to output debug, warning, and error messages for troubleshooting and performance monitoring.

---

## Future Improvements

* Implement a more robust tracker (e.g., DeepSORT) for improved object tracking.
* Add support for live camera feeds in addition to video files.
* Enhance drowning detection with machine learning models trained on drowning-specific datasets.
* Integrate environmental factors (e.g., water depth, lighting conditions).
* Optimize performance for real-time processing on lower-end hardware.

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
