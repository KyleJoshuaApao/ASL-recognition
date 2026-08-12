import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, Response, jsonify
from model import build_model
from data_collection import mediapipe_detection_new, draw_styled_landmarks_new, extract_keypoints_from_results

app = Flask(__name__)

# Global variables for prediction state
current_prediction = "Waiting for model..."
camera = None

# Model Setup
DATA_PATH = os.path.join('MP_Data')
alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]
numbers = [str(i) for i in range(1, 10)]
actions = np.array(alphabet + numbers)

model = None
if len(actions) > 0:
    model = build_model(len(actions))
    try:
        model.load_state_dict(torch.load('action.pth'))
        model.eval()
        print("Model loaded successfully.")
    except:
        model = None
        current_prediction = "Please train model (action.pth not found)"
else:
    current_prediction = "Please process dataset first"

def generate_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)
    
    sequence = []
    predictions = []
    threshold = 0.40  # Lower threshold so predictions actually show
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # We need to mirror the frame so the webcam acts like a mirror
        frame = cv2.flip(frame, 1)
        
        image, pose_result, hand_result = mediapipe_detection_new(frame)
        draw_styled_landmarks_new(image, pose_result, hand_result)
        
        if model is not None:
            keypoints = extract_keypoints_from_results(pose_result, hand_result)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = model(input_tensor)
                    probs = F.softmax(out, dim=1)
                    res = probs[0].numpy()
                
                top_idx = int(np.argmax(res))
                top_conf = float(res[top_idx])
                top_label = actions[top_idx]
                
                predictions.append(top_idx)
                predictions = predictions[-10:]  # Keep last 10
                
                # Show the prediction if confidence is above threshold
                # Use a majority vote across last 5 predictions to smooth jitter
                if len(predictions) >= 5:
                    from collections import Counter
                    most_common_idx = Counter(predictions[-5:]).most_common(1)[0][0]
                    most_common_conf = float(res[most_common_idx])
                    if most_common_conf > threshold:
                        current_prediction = f"{actions[most_common_idx]} ({most_common_conf*100:.0f}%)"
                    else:
                        current_prediction = f"Uncertain... ({top_label} {top_conf*100:.0f}%)"
                else:
                    current_prediction = f"Reading... ({top_label} {top_conf*100:.0f}%)"
                
        ret, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', actions=actions.tolist() if hasattr(actions, 'tolist') else actions)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def get_prediction():
    return jsonify({"prediction": current_prediction})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
