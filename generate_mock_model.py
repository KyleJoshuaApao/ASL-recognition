import os
import torch
from model import build_model

def create_mock_model():
    alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    numbers = [str(i) for i in range(1, 10)]
    actions = alphabet + numbers
    data_path = os.path.join('MP_Data')
    
    # 1. Create dataset folders
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)
            
    print(f"Created vocabulary folders in {data_path} for: {actions}")
    
    # 2. Build model and save weights
    model = build_model(len(actions))
    torch.save(model.state_dict(), 'action.pth')
    print("Successfully generated mock pre-trained model: action.pth")

if __name__ == '__main__':
    create_mock_model()
