# Sign Language Recognition

This project uses MediaPipe for hand and pose tracking and an LSTM neural network (via TensorFlow/Keras) for recognizing dynamic sign language gestures.

## Setup Instructions

1. **Install Requirements**:
   Make sure you have python installed, and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect Data**:
   Run the data collection script to record your own dataset. It will record 30 sequences (videos) of 30 frames each for the actions `hello`, `thanks`, and `iloveyou`.
   ```bash
   python data_collection.py
   ```
   Follow the on-screen prompts to perform the signs.

3. **Train the Model**:
   After collecting data, train the LSTM model by running:
   ```bash
   python train.py
   ```
   This will output an `action.h5` file containing the trained weights.

4. **Real-time Inference**:
   Once the model is trained, you can run the inference script to predict signs in real-time from your webcam:
   ```bash
   python inference.py
   ```
   Press `q` to quit the application.
