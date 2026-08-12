import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Model Paths
HAND_MODEL_PATH = 'hand_landmarker.task'
POSE_MODEL_PATH = 'pose_landmarker.task'

# Configuration for Landmarkers
base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand,
                                            num_hands=2)
hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

# We draw connections manually to avoid depending on mp.solutions
# Since mp.solutions failed earlier, let's draw connections manually to be safe.
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

def extract_keypoints_from_results(pose_result, hand_result):
    # Pose: 33 landmarks * 4 (x,y,z,visibility)
    # The new API has x,y,z,presence,visibility
    if pose_result and pose_result.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33*4)
        
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if hand_result and hand_result.hand_landmarks:
        for idx, handedness in enumerate(hand_result.handedness):
            # handedness[0].category_name is 'Left' or 'Right'
            hand_type = handedness[0].category_name
            landmarks = hand_result.hand_landmarks[idx]
            arr = np.array([[res.x, res.y, res.z] for res in landmarks]).flatten()
            if hand_type == 'Left':
                lh = arr
            elif hand_type == 'Right':
                rh = arr
                
    return np.concatenate([pose, lh, rh])

def mediapipe_detection_new(image):
    # Convert the image to MediaPipe format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Detect
    hand_result = hand_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)
    
    return image, pose_result, hand_result

def draw_styled_landmarks_new(image, pose_result, hand_result):
    h, w, _ = image.shape
    
    # Draw Pose
    if pose_result and pose_result.pose_landmarks:
        for landmarks in pose_result.pose_landmarks:
            for connection in POSE_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                start_pt = (int(start.x * w), int(start.y * h))
                end_pt = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_pt, end_pt, (80,22,10), 2)
            for lm in landmarks:
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (80,44,121), -1)
                
    # Draw Hands
    if hand_result and hand_result.hand_landmarks:
        for landmarks in hand_result.hand_landmarks:
            for connection in HAND_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                start_pt = (int(start.x * w), int(start.y * h))
                end_pt = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_pt, end_pt, (121,22,76), 2)
            for lm in landmarks:
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (121,44,250), -1)

# Sequence length
sequence_length = 30
