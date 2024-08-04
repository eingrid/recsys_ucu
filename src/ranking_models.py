import torch
import torch.nn as nn

class PairwiseRankingModel(nn.Module):
    def __init__(self):
        super(PairwiseRankingModel, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(3 + 19, 64)  # User features (7) + Film1 features (19)
        self.fc2 = nn.Linear(64 + 19, 32)  # Add Film2 features (19)
        self.fc3 = nn.Linear(32, 1)  # Output

    def forward(self, user_features, film1_features, film2_features):
        # Concatenate user features and film1 features
        x = torch.cat([user_features, film1_features], dim=1)
        x = torch.relu(self.fc1(x))
        # Concatenate with film2 features
        x = torch.cat([x, film2_features], dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_pairwise_model(path="../../artifacts/pairwise_ranking_model.pth", device='cpu'):
    model = PairwiseRankingModel().to(device)  # Instantiate the model
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model