import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join('MP_Data')
sequence_length = 30

def load_data():
    if not os.path.exists(DATA_PATH):
        raise ValueError(f"Data path {DATA_PATH} not found. Please collect or process data first.")
        
    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    label_map = {label:num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        sequence_dirs = [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))]
        
        for sequence_dir in sequence_dirs:
            window = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(action_dir, sequence_dir, f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    res = np.load(npy_path)
                    window.append(res)
                else:
                    window.append(np.zeros(258))
                    
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
            
    X = np.array(sequences)
    y = np.array(labels).astype(np.int64)
    return X, y, actions

if __name__ == '__main__':
    X, y, actions = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print(f"Detected actions: {actions}")


