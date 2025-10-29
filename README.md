

# Object and Face Detection for Blind Assistance

## Overview

This project leverages **YOLO (You Only Look Once)** for real-time **object detection** and a custom **face detection model** to identify and recognize the faces of your relatives. The application is designed specifically to assist blind or visually impaired individuals by identifying objects and people in their immediate surroundings.

### Key Features:

* **Object Detection**: Uses the YOLO algorithm to detect and identify a variety of objects in the environment.
* **Face Recognition**: Custom-trained model to detect and recognize faces of specific people (e.g., family members or friends).
* **Real-Time Assistance**: A real-time system that can process live video streams from a camera, making it ideal for hands-free use.
* **Blind Assistance**: Provides valuable feedback to blind or visually impaired users about what is in front of them—helping identify objects and recognize people in real-time.

## Technologies

* **Python** (primary language)
* **YOLO (You Only Look Once)** for object detection
* **OpenCV** for real-time video capture and image processing
* **TensorFlow** or **Keras** for custom face recognition models
* **NumPy** for numerical operations
* **dlib** for facial landmark detection (if used)
* **Pre-trained face recognition model** for recognizing relatives

## Installation

### Prerequisites

Ensure the following dependencies are installed:

* Python 3.x
* pip (Python package installer)

### Install Dependencies

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/face-object-detection.git
   cd face-object-detection
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the necessary Python libraries:

   pip install -r requirements.txt
   

   Your `requirements.txt` file should look something like this:

   opencv-python
   tensorflow
   torch
   numpy
   matplotlib
   dlib
   

4. Download the YOLO weights and configuration files (available at the [YOLO website](https://pjreddie.com/darknet/yolo/)) and place them in the `models/` directory.

5. Pre-train or add the face recognition model for your relatives in the `models/faces/` directory. If you’re using a pre-trained model like **FaceNet** or **dlib**, ensure you have the model file available.


 1. Detect Faces in Real-Time

To use the face recognition model with your webcam, use the following command:


python real_time_face_recognition.py

This script will capture video from your webcam, recognize faces that have been pre-trained (your relatives), and output their names when detected.

### 2. Object Detection in Real-Time

For real-time object detection using the YOLO algorithm with your webcam, run:


python real_time_object_detection.py


This will display a live feed from your camera with detected objects (like people, vehicles, animals) outlined with bounding boxes.

### 3. Detect Objects and Faces in a Video File

To detect both faces and objects in a video file:


python detect_objects_faces.py --video path_to_video.mp4


This will process the video and display identified objects and faces in each frame.

### 4. Customizing Face Recognition for Your Relatives

To add new faces of your relatives to the system:

1. Collect images of your relatives in different poses and lighting.
2. Run the `train_faces.py` script to pre-train a model for face recognition.


python train_faces.py --images path_to_images_directory


This will save the trained model in the `models/faces/` directory, which can then be used for recognition.

### 5. Voice Feedback for Blind Assistance

For real-time voice feedback on detected objects and faces, the system can be extended to include text-to-speech features. You can integrate libraries like **pyttsx3** to provide auditory cues for detected objects and recognized faces.

## Folder Structure

face-object-detection/
│
├── models/                # Folder for pre-trained models (YOLO, faces, etc.)
│   ├── yolo/              # YOLO weights and config files
│   └── faces/             # Trained models for face recognition
├── images/                # Folder for storing input/output images
├── video/                 # Folder for video files
├── detect_objects_faces.py # Script for detecting objects and faces in videos
├── real_time_face_recognition.py # Real-time face recognition from webcam
├── real_time_object_detection.py # Real-time object detection from webcam
├── train_faces.py         # Script to train face recognition model
├── requirements.txt       # File with the dependencies
└── README.md              # This file




