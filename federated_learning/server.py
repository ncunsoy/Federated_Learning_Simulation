"""
Federated Learning Server
TCP Socket-based implementation
"""

import socket
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
from utils import SimpleNN, load_mnist, serialize_model, deserialize_model, serialize_message, deserialize_message

class FederatedServer:
    def __init__(self, num_clients, num_rounds, device='cpu', port=5000):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.device = device
        self.port = port
        
        # Model
        self.global_model = SimpleNN().to(device)
        
        # Network
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen(num_clients)
        
        # Connected clients
        self.clients = []
        
        # History
        self.history = {
            'rounds': [],
            'accuracy': [],
            'loss': []
        }
        
        print(f"\n{'*'*70}")
        print(f"FL Server Started on localhost:{port}")
        print(f"Waiting for {num_clients} clients to connect...")
        print(f"{'*'*70}\n")
    
    def wait_for_clients(self):
        """Wait for all clients to connect"""
        for i in range(self.num_clients):
            client_sock, addr = self.server_socket.accept()
            self.clients.append(client_sock)
            print(f" Client {i} connected from {addr}")
        
        print(f"\nAll {self.num_clients} clients connected!\n")
    
    def broadcast_model(self):
        """Send global model to all clients"""
        model_data = serialize_model(self.global_model.state_dict())
        
        for i, client_sock in enumerate(self.clients):
            # Send model size first
            model_size = len(model_data)
            client_sock.sendall(model_size.to_bytes(4, 'big'))
            
            # Send model data
            client_sock.sendall(model_data)
            print(f"   Sent model to Client {i} ({model_size} bytes)")
    
    def collect_updates(self):
        """Collect model updates from all clients"""
        client_updates = []
        client_sizes = []
        
        for i, client_sock in enumerate(self.clients):
            # Receive update size
            size_data = client_sock.recv(4)
            update_size = int.from_bytes(size_data, 'big')
            
            # Receive update data
            update_data = b''
            while len(update_data) < update_size:
                chunk = client_sock.recv(min(4096, update_size - len(update_data)))
                if not chunk:
                    break
                update_data += chunk
            
            # Deserialize
            update_info = deserialize_message(update_data)
            client_updates.append(update_info['model'])
            client_sizes.append(update_info['data_size'])
            
            print(f"  ← Received update from Client {i} ({update_size} bytes, {update_info['data_size']} samples)")
        
        return client_updates, client_sizes
    
    def aggregate(self, client_updates, client_sizes):
        """
        FedAvg: Weighted average aggregation
        w_global = Σ (n_k / n_total) × w_k
        """
        global_dict = self.global_model.state_dict()
        total_size = sum(client_sizes)
        
        # Initialize with zeros
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Weighted sum
        for client_params, size in zip(client_updates, client_sizes):
            weight = size / total_size
            for key in global_dict.keys():
                global_dict[key] += client_params[key] * weight
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        print(f"   Aggregated {len(client_updates)} client updates (FedAvg)")
    
    def evaluate(self, test_loader):
        """Evaluate global model"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def train(self):
        """Main training loop"""
        # Load test data
        _, test_dataset = load_mnist()
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        print(f"\n{'*'*70}")
        print(f"Starting Federated Learning ({self.num_rounds} rounds)")
        print(f"{'*'*70}\n")
        
        for round_num in range(self.num_rounds):
            print(f"Round {round_num + 1}/{self.num_rounds}")
            print("-" * 70)
            
            # 1. Broadcast global model
            self.broadcast_model()
            
            # 2. Signal clients to start training
            for client_sock in self.clients:
                msg = serialize_message({'command': 'train'})
                client_sock.sendall(len(msg).to_bytes(4, 'big'))
                client_sock.sendall(msg)
            
            # 3. Collect updates
            client_updates, client_sizes = self.collect_updates()
            
            # 4. Aggregate (FedAvg)
            self.aggregate(client_updates, client_sizes)
            
            # 5. Evaluate
            accuracy, loss = self.evaluate(test_loader)
            self.history['rounds'].append(round_num + 1)
            self.history['accuracy'].append(accuracy)
            self.history['loss'].append(loss)
            
            print(f" Global Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}\n")
        
        print(f"\n{'*'*70}")
        print(f"Training Completed!")
        print(f"Final Accuracy: {self.history['accuracy'][-1]:.2f}%")
        print(f"{'*'*70}\n")
        
        # Save results
        import json
        with open(f'fl_results_{self.num_clients}clients.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f" Results saved to fl_results_{self.num_clients}clients.json\n")
    
    def shutdown(self):
        """Close all connections"""
        # Signal clients to shutdown
        for client_sock in self.clients:
            try:
                msg = serialize_message({'command': 'shutdown'})
                client_sock.sendall(len(msg).to_bytes(4, 'big'))
                client_sock.sendall(msg)
                client_sock.close()
            except:
                pass
        
        self.server_socket.close()
        print("Server shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients (3 or 5)')
    parser.add_argument('--rounds', type=int, default=10, help='Number of training rounds')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    server = FederatedServer(
        num_clients=args.num_clients,
        num_rounds=args.rounds,
        device=device,
        port=args.port
    )
    
    try:
        server.wait_for_clients()
        server.train()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        server.shutdown()

if __name__ == "__main__":
    main()