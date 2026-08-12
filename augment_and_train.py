import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import build_model
import glob

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence_length = 30

def augment_sequence(sequence):
    # Add random noise
    noise = np.random.normal(0, 0.02, sequence.shape)
    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    
    aug_seq = (sequence + noise) * scale
    return aug_seq

def load_and_augment_data():
    X, y = [], []
    for action_idx, action in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path): continue
        
        # Load all existing sequences for this action
        sequences = []
        for seq_folder in os.listdir(action_path):
            seq_path = os.path.join(action_path, seq_folder)
            if not os.path.isdir(seq_path): continue
            
            # Load frames
            frames = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    frames.append(np.load(frame_path))
                else:
                    frames.append(np.zeros(258))
                    
            if len(frames) == sequence_length:
                sequences.append(np.array(frames))
                
        # Now augment them to create 200 sequences per action
        print(f"Action {action} has {len(sequences)} base sequences. Augmenting...")
        if len(sequences) > 0:
            for _ in range(200):
                base_seq = sequences[np.random.randint(0, len(sequences))]
                X.append(augment_sequence(base_seq))
                y.append(action_idx)
                
    return np.array(X), np.array(y)

def train_robust_model():
    print("Generating large synthetic dataset from base WLASL videos...")
    X, y = load_and_augment_data()
    
    if len(X) == 0:
        print("No data found to augment.")
        return
        
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = build_model(len(actions))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training robust model on augmented dataset...")
    model.train()
    for epoch in range(150):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/150], Loss: {running_loss/len(dataloader):.4f}, Acc: {100*correct/total:.2f}%")
            
    torch.save(model.state_dict(), 'action.pth')
    print("Robust model saved as action.pth!")

if __name__ == '__main__':
    train_robust_model()
