import cv2
import numpy as np
import mediapipe as mp
import os
import torch
import torch.nn.functional as F
from model import build_model
from data_collection import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import time

DATA_PATH = os.path.join('MP_Data')
if not os.path.exists(DATA_PATH):
    raise ValueError(f"Data path {DATA_PATH} not found. Please collect or process data first.")
actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

sequence = []
sentence = []
predictions = []
threshold = 0.5

# Load model
model = build_model(len(actions))
try:
    model.load_state_dict(torch.load('action.pth'))
    model.eval()
    print("Model weights loaded successfully.")
except:
    print("Could not load model weights. Please train the model first.")

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            # Convert to PyTorch tensor
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0) # add batch dim
            
            with torch.no_grad():
                out = model(input_tensor)
                probs = F.softmax(out, dim=1)
                res = probs[0].numpy()
                
            predictions.append(np.argmax(res))
            
            # Rendering logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
