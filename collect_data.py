import cv2
import numpy as np
import os
import time
from data_collection import mediapipe_detection_new, draw_styled_landmarks_new, extract_keypoints_from_results, DATA_PATH, sequence_length

# Actions that we try to detect (ASL Alphabet A-Z and 1-9)
alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]
numbers = [str(i) for i in range(1, 10)]
actions = np.array(alphabet + numbers)
# Thirty videos worth of data
no_sequences = 15

def collect():
    print("Setting up folders for data collection...")
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue

                # Mirror image so it's easier to follow
                frame = cv2.flip(frame, 1)

                image, pose, hand = mediapipe_detection_new(frame)
                draw_styled_landmarks_new(image, pose, hand)

                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Wait 2 seconds before each video so you can get in position
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints_from_results(pose, hand)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Data collection cancelled.")
                    return
                    
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete! You can now run train.py")

if __name__ == '__main__':
    collect()
