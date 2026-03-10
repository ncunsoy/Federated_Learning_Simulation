"""
Federated Learning Client
TCP Socket-based implementation
"""

import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import argparse
import time
from utils import SimpleNN, load_mnist, create_non_iid_data, serialize_model, deserialize_model, serialize_message, deserialize_message

class FederatedClient:
    def __init__(self, client_id, num_clients, local_epochs=2, device='cpu', server_host='localhost', server_port=5000):
        self.client_id = client_id
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.device = device
        
        # Model
        self.model = SimpleNN().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data
        train_dataset, _ = load_mnist()
        client_indices = create_non_iid_data(train_dataset, num_clients)
        self.train_data = Subset(train_dataset, client_indices[client_id])
        self.train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        
        # Network
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_host = server_host
        self.server_port = server_port
        
        print(f"\n{'='*70}")
        print(f"Client {client_id} initialized")
        print(f"Training samples: {len(self.train_data)}")
        print(f"{'='*70}\n")
    
    def connect_to_server(self):
        """Connect to FL server"""
        print(f"Connecting to server at {self.server_host}:{self.server_port}...")
        self.socket.connect((self.server_host, self.server_port))
        print(f" Connected to server!\n")
    
    def receive_model(self):
        """Receive global model from server"""
        # Receive model size
        size_data = self.socket.recv(4)
        model_size = int.from_bytes(size_data, 'big')
        
        # Receive model data
        model_data = b''
        while len(model_data) < model_size:
            chunk = self.socket.recv(min(4096, model_size - len(model_data)))
            if not chunk:
                break
            model_data += chunk
        
        # Load model
        model_state = deserialize_model(model_data)
        self.model.load_state_dict(model_state)
        
        print(f"   Received global model ({model_size} bytes)")
    
    def train_local(self):
        """Train on local data"""
        self.model.train()
        total_loss = 0
        
        print(f"   Training locally for {self.local_epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(self.train_loader)
        
        training_time = time.time() - start_time
        avg_loss = total_loss / self.local_epochs
        
        print(f"   Training complete (Loss: {avg_loss:.4f}, Time: {training_time:.2f}s)")
        
        return avg_loss
    
    def send_update(self):
        """Send model update to server"""
        update_info = {
            'model': self.model.state_dict(),
            'data_size': len(self.train_data)
        }
        
        update_data = serialize_message(update_info)
        
        # Send size then data
        self.socket.sendall(len(update_data).to_bytes(4, 'big'))
        self.socket.sendall(update_data)
        
        print(f"   Sent update to server ({len(update_data)} bytes)\n")
    
    def run(self):
        """Main client loop"""
        while True:
            try:
                # 1. First receive model
                self.receive_model()
                
                # 2. Then receive train command
                size_data = self.socket.recv(4)
                if not size_data:
                    break
                
                msg_size = int.from_bytes(size_data, 'big')
                msg_data = self.socket.recv(msg_size)
                command = deserialize_message(msg_data)
                
                if command.get('command') == 'shutdown':
                    print("Shutdown signal received.")
                    break
                
                # 3. Train locally
                self.train_local()
                
                # 4. Send update
                self.send_update()
                
            except Exception as e:
                print(f"Error in training loop: {e}")
                break
        
        self.socket.close()
        print(f"Client {self.client_id} disconnected.\n")

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID (0, 1, 2, ...)')
    parser.add_argument('--num_clients', type=int, default=3, help='Total number of clients')
    parser.add_argument('--epochs', type=int, default=2, help='Local training epochs')
    parser.add_argument('--server_host', type=str, default='localhost', help='Server host')
    parser.add_argument('--server_port', type=int, default=5000, help='Server port')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    client = FederatedClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        local_epochs=args.epochs,
        device=device,
        server_host=args.server_host,
        server_port=args.server_port
    )
    
    try:
        client.connect_to_server()
        client.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Client shutdown complete.")

if __name__ == "__main__":
    main()