import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, num_actions):
        super(SignLanguageModel, self).__init__()
        # Input shape per frame is 258
        self.lstm1 = nn.LSTM(input_size=258, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        
        self.fc1 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_actions)
        
    def forward(self, x):
        # x is (batch_size, seq_len=30, input_size=258)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        
        # We only need the output of the last time step for classification
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # Note: In PyTorch, nn.CrossEntropyLoss automatically applies Softmax,
        # so we don't apply it here in the forward pass.
        return out

def build_model(actions_shape):
    return SignLanguageModel(actions_shape)
