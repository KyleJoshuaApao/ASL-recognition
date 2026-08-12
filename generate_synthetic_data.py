import os
import numpy as np

def generate_synthetic_data():
    alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    numbers = [str(i) for i in range(1, 10)]
    actions = alphabet + numbers
    
    no_sequences = 15
    sequence_length = 30
    DATA_PATH = os.path.join('MP_Data')
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    print("Generating synthetic dataset (fast mode) to bypass manual video collection...")
    
    # MediaPipe extracts 258 features per frame (33*4 + 21*3 + 21*3)
    feature_size = 258
    
    for idx, action in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)
            
        # Create a distinct synthetic "pose" signature for this action
        # so the neural network can easily learn to separate them.
        np.random.seed(idx) 
        base_signature = np.random.rand(feature_size)
        
        for sequence in range(no_sequences):
            seq_path = os.path.join(action_path, str(sequence))
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
                
            for frame_num in range(sequence_length):
                # Add slight noise to simulate movement
                noise = np.random.normal(0, 0.05, feature_size)
                frame_data = base_signature + noise
                
                npy_path = os.path.join(seq_path, str(frame_num) + '.npy')
                np.save(npy_path, frame_data)
                
    print("Synthetic dataset successfully created in 'MP_Data' folder!")
    print("You can now run 'python train.py' to train the model quickly.")

if __name__ == '__main__':
    generate_synthetic_data()
