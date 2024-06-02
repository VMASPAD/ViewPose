# Real-Time Object Detection with YOLOv8

This Python script demonstrates the use of the YOLOv8 object detection model for real-time object detection using a webcam feed. It utilizes the Ultralytics library for loading and running the YOLO model, and OpenCV for video capture and display.

## Requirements

- Python 3.x
- Ultralytics (YOLOv8) library: `pip install ultralytics`
- OpenCV: `pip install opencv-python`
- Pre-trained YOLOv8 model: `POSE.pt` (included in the repository)

## Usage

1. Clone the repository or copy the code into a Python script file.
2. Ensure that the `POSE.pt` model file is present in the same directory as the script.
3. Run the script using the following command:

```
python script_name.py
```

4. The script will start capturing video from the default webcam (index 0).
5. The real-time object detection results will be displayed in a window titled "View".
6. Press the "Esc" key to exit the program.

## Code Explanation

```python
# Import the required libraries
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLO model
model = YOLO("POSE.pt")

# Initialize video capture
cap = cv2.VideoCapture(1)  # 1 for external webcam, 0 for built-in webcam

# Loop for real-time detection
while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Run the YOLO model on the frame
    results = model.predict(frame, imgsz=340)

    # Plot the detected objects on the frame
    annotations = results[0].plot()

    # Display the annotated frame
    cv2.imshow("View", annotations)

    # Exit condition (press 'Esc' to exit)
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
```

The script follows these steps:

1. Import the required libraries: `ultralytics` for the YOLOv8 model and `cv2` for video capture and display.
2. Load the pre-trained YOLOv8 model from the `POSE.pt` file.
3. Initialize the video capture object using `cv2.VideoCapture(1)` for an external webcam or `cv2.VideoCapture(0)` for a built-in webcam.
4. Enter a loop for real-time object detection:
   - Read a frame from the webcam using `cap.read()`.
   - Run the YOLOv8 model on the frame using `model.predict(frame, imgsz=340)`. The `imgsz=340` parameter sets the input image size for the model.
   - Plot the detected objects on the frame using `results[0].plot()`.
   - Display the annotated frame in a window titled "View" using `cv2.imshow("View", annotations)`.
   - Check for the "Esc" key press to exit the loop and program.
5. Release the video capture and close all windows using `cap.release()` and `cv2.destroyAllWindows()`.

Note: Make sure to have the pre-trained `POSE.pt` model file in the same directory as the script.
