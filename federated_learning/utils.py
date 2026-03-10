"""
Utility functions for Federated Learning
Model definition, data loading, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pickle

# MODEL 
class SimpleNN(nn.Module):
    """Simple Neural Network for MNIST"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DATA 
def load_mnist():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Üst klasördeki data'yı kullan
    data_path = r'D:\Desktop\Federated_Learning\data'
    
    train_dataset = datasets.MNIST(data_path, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=False, transform=transform)
    
    return train_dataset, test_dataset

def create_non_iid_data(dataset, num_clients):
    """
    Create Non-IID data distribution
    Each client gets different digits (with overlap)
    """
    # Group by labels
    digit_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        digit_indices[label].append(idx)
    
    client_data = [[] for _ in range(num_clients)]
    
    # Automatic distribution: each client gets 2-3 digits with overlap
    digits_per_client = 2 if num_clients > 5 else 3
    
    for client_id in range(num_clients):
        # Each client gets sequential digits with wraparound
        start_digit = (client_id * 10 // num_clients) % 10
        for i in range(digits_per_client):
            digit = (start_digit + i) % 10
            client_data[client_id].extend(digit_indices[digit])
    
    return client_data

# SERIALIZATION 
def serialize_model(model_state_dict):
    """Serialize model parameters for network transmission"""
    return pickle.dumps(model_state_dict)

def deserialize_model(data):
    """Deserialize model parameters from network"""
    return pickle.loads(data)

def serialize_message(message):
    """Serialize any message (dict, list, etc.)"""
    return pickle.dumps(message)

def deserialize_message(data):
    """Deserialize message"""
    return pickle.loads(data)