# Federated Learning over Networks

**Network Optimization for Decentralized Model Training**

A Python implementation of Federated Learning using raw TCP sockets and PyTorch, simulating the FedAvg algorithm across multiple clients with non-IID data distribution.

> Full technical report: [report.pdf](report.pdf)

---

## Overview

This project implements a federated learning system where a central server coordinates model training across distributed clients — without ever collecting raw data. Each client trains locally on its private MNIST partition and sends only the model weights back to the server for aggregation.

```
Global Model
    │
    ├──► Client 0 ──► Local Training ──► Weight Update ──┐
    ├──► Client 1 ──► Local Training ──► Weight Update ──┼──► FedAvg ──► New Global Model
    └──► Client N ──► Local Training ──► Weight Update ──┘
```

## Project Structure

```
Federated_Learning/
├── federated_learning/
│   ├── server.py              # FL server (FedAvg aggregation)
│   ├── client.py              # FL client (local training)
│   ├── utils.py               # Model, data loading, serialization
│   └── results/
│       ├── client3/           # 3-client experiment results
│       ├── client5/           # 5-client experiment results
│       └── client10/          # 10-client experiment results
├── centralized/
│   ├── centralized_baseline.py  # Centralized training for comparison
│   └── centralized_baseline.png
├── data/
│   └── MNIST/ 
│       └── raw/    # MNIST dataset
└── report.pdf                 # Full technical report
```

## Model & Data

- **Model:** SimpleNN — 3-layer fully connected network (784 → 128 → 64 → 10)
- **Dataset:** MNIST (60,000 training / 10,000 test images)
- **Distribution:** Non-IID — each client receives a skewed subset of digit classes to simulate real-world heterogeneity

## Requirements

```
Python 3.8+
PyTorch
torchvision
```

Install dependencies:

```bash
pip install torch torchvision
```

## Usage

### Federated Training

Open a separate terminal for each process. Start the server first, then launch the clients.

**Terminal 1 — Server:**
```bash
cd federated_learning
python server.py --num_clients 3 --rounds 10 --port 5000
```

**Terminals 2, 3, 4 — Clients (one per terminal):**
```bash
python client.py --client_id 0 --num_clients 3
python client.py --client_id 1 --num_clients 3
python client.py --client_id 2 --num_clients 3
```

**Server arguments:**

| Argument | Default | Description |
|---|---|---|
| `--num_clients` | 3 | Number of clients to wait for |
| `--rounds` | 10 | Number of FL communication rounds |
| `--port` | 5000 | TCP port |

**Client arguments:**

| Argument | Default | Description |
|---|---|---|
| `--client_id` | required | Client index (0-based) |
| `--num_clients` | 3 | Total number of clients |
| `--epochs` | 2 | Local training epochs per round |
| `--server_host` | localhost | Server address |
| `--server_port` | 5000 | Server port |

### Centralized Baseline

```bash
cd centralized
python centralized_baseline.py
```

## Results

Global model accuracy (%) over 10 FL rounds:

| Round | 3 Clients | 10 Clients |
|:-----:|:---------:|:----------:|
| 1     | 22.77     | 50.96      |
| 2     | 71.69     | 71.68      |
| 3     | 78.58     | 78.02      |
| 4     | 86.65     | 79.70      |
| 5     | 91.38     | 83.88      |
| 6     | 91.12     | 84.84      |
| 7     | 91.63     | 86.43      |
| 8     | 91.77     | 87.35      |
| 9     | 92.99     | 88.80      |
| 10    | **93.65** | **89.08**  |

The 3-client configuration converges faster and achieves higher final accuracy. With 10 clients, data is partitioned across narrower digit class ranges, increasing statistical heterogeneity and slowing convergence — consistent with the weight divergence effect described in the report.

Network traffic measured via Wireshark:

| Config | Total Traffic | Duration | Avg Throughput |
|--------|:-------------:|:--------:|:--------------:|
| 3 clients | 26 MB | 12m 29s | 281 kbps |
| 10 clients | 88 MB | 9m 34s | 1,227 kbps |

Each TCP conversation transfers ~4 MB per direction per round, corresponding to the serialized SimpleNN parameter payload.

## How It Works

1. **Broadcast (Downlink):** Server serializes the global model and sends it to all clients over TCP.
2. **Local Training:** Each client trains for 2 epochs using SGD on its private data partition.
3. **Upload (Uplink):** Clients send their updated weights and local dataset size back to the server.
4. **Aggregation (FedAvg):** Server computes the weighted average:

$$w_{global} = \sum_{k=1}^{K} \frac{n_k}{n_{total}} w_k$$

5. **Evaluation:** Server tests the new global model on the held-out MNIST test set.

## References

1. McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629), 2017
2. Kairouz et al., [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977), 2021
3. Konečný et al., [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492), 2017
4. Li et al., [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127), 2020
5. Bonawitz et al., [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046), 2019
6. Zhao et al., [Federated Learning with Non-IID Data](https://arxiv.org/abs/1806.00582), 2018

---

