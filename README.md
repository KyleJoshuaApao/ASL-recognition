# ASL Alphabet & Number Recognition System

A real-time American Sign Language (ASL) recognition system. It uses **MediaPipe** for hand and pose landmark extraction and a **PyTorch LSTM Neural Network** to predict the ASL Alphabet (**A to Z**) and Numbers (**1 to 9**) directly from a live camera feed.

The repository comes pre-loaded with a trained **`action.pth`** model, so other users can run it instantly without needing to download large datasets or train it themselves.

---

## 🚀 How to Run (For New Users)

You can run the application immediately using the pre-trained model:

### 1. Install Requirements
Ensure you have Python installed, then open your terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
Launch the Flask web server:
```bash
python app.py
```

### 3. Open in Browser
Open your web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Place your hand in front of your webcam, and the system will start predicting your ASL signs in real-time!

---

## 🛠️ Developer Pipeline (Optional)

If you want to customize the signs or re-train the model from scratch, you can use these tools:

### 1. Data Collection
To collect new coordinate data from your webcam:
```bash
python collect_data.py
```
This will record sequences of keypoint arrays and save them in the `MP_Data/` directory.

### 2. Model Training
To train the neural network on the current keypoints in `MP_Data/`:
```bash
python train.py
```
This will compile the model and overwrite the **`action.pth`** file with your newly trained weights.

### 3. Desktop Inference (No Web UI)
To run predictions directly in a raw OpenCV window:
```bash
python inference.py
```
