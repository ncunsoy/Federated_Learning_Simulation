"""
Centralized Training Baseline
For comparison with Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# MODEL
class SimpleNN(nn.Module):
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

# CENTRALIZED TRAINING
def train_centralized(num_epochs=20, device='cpu'):
    print("\n" + "="*70)
    print("CENTRALIZED TRAINING (Baseline)")
    print("="*70 + "\n")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Model, optimizer, loss
    model = SimpleNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'epochs': [],
        'accuracy': [],
        'loss': []
    }
        
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        
        history['epochs'].append(epoch + 1)
        history['accuracy'].append(accuracy)
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print(f"\nFinal Accuracy: {history['accuracy'][-1]:.2f}%\n")
    
    return history

# VISUALIZATION
def plot_centralized_results(history):
    """Plot training results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history['epochs'], history['accuracy'], 'b-o', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Centralized Training - Accuracy', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['epochs'], history['loss'], 'r-o', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Centralized Training - Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('centralized_baseline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: centralized_baseline.png")
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train centralized model
    history = train_centralized(num_epochs=20, device=device)
    
    # Plot results
    plot_centralized_results(history)
