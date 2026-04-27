# Federated Learning Simulation

TCP soket tabanlı **Federated Learning (FL)** simülasyonu. MNIST veri seti üzerinde, merkezi öğrenme (baseline) ile karşılaştırmalı olarak farklı istemci sayılarıyla (3, 5, 10) dağıtık model eğitimi gerçekleştirilmektedir.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Proje Yapısı](#proje-yapısı)
- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Deney Sonuçları](#deney-sonuçları)
- [Non-IID Veri Dağılımı](#non-iid-veri-dağılımı)

---

## Proje Hakkında

Bu proje, **Federated Learning**'in temel ilkelerini TCP soket haberleşmesi üzerine inşa ederek somutlaştırmaktadır:

- İstemciler yalnızca kendi yerel verileriyle eğitim yapar; ham veri sunucuya gönderilmez.
- Sunucu, istemcilerden gelen model güncellemelerini **FedAvg** algoritmasıyla birleştirir.
- Merkezi eğitim (centralized baseline) ile federated eğitim sonuçları karşılaştırılır.

### Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|---------------|
| Python 3.x | Ana programlama dili |
| PyTorch | Sinir ağı modeli ve eğitim |
| torchvision | MNIST veri seti yükleme |
| socket (stdlib) | TCP tabanlı istemci-sunucu iletişimi |
| pickle (stdlib) | Model parametrelerinin serileştirilmesi |
| matplotlib | Sonuçların görselleştirilmesi |

---

## Proje Yapısı

```
Federated_Learning_Simulation/
│
├── data/                          # MNIST veri seti
│   └── MNIST/
│
├── federated_learning/            # Federated Learning implementasyonu
│   ├── server.py                  # FL Sunucusu (FedAvg aggregation)
│   ├── client.py                  # FL İstemcisi (yerel eğitim)
│   ├── utils.py                   # Model tanımı, veri yükleme, yardımcı fonksiyonlar
│   ├── client3/                   # 3 istemcili deney sonuçları
│   │   └── fl_results_3clients.json
│   ├── client5/                   # 5 istemcili deney sonuçları
│   │   └── fl_results_5clients.json
│   └── client10/                  # 10 istemcili deney sonuçları
│       └── fl_results_10clients.json
│
└── centralized/                   # Merkezi eğitim (karşılaştırma için)
    ├── centralized_baseline.py
    └── centralized_baseline.png
```

---

## Mimari

### Model: SimpleNN

MNIST sınıflandırması için 3 katmanlı tam bağlı ağ:

```
Giriş (784)  →  FC1 (128)  →  ReLU  →  FC2 (64)  →  ReLU  →  FC3 (10)
```

### FedAvg Algoritması

Her round için:

1. **Yayın (Broadcast):** Sunucu global modeli tüm istemcilere gönderir.
2. **Yerel Eğitim:** Her istemci kendi veri alt kümesiyle modeli günceller.
3. **Güncelleme Toplama:** Sunucu güncellenmiş ağırlıkları toplar.
4. **Ağırlıklı Ortalama:** Sunucu istemci veri boyutuna göre ağırlıklı ortalama alır:

```
w_global = Σ (n_k / n_total) × w_k
```

5. **Değerlendirme:** Sunucu global modeli test seti üzerinde değerlendirir.

### İletişim Protokolü

```
Sunucu                          İstemci
  |  ─── Global Model ────────►  |
  |  ─── "train" komutu ──────►  |
  |  ◄─── Model Güncellemesi ──  |
  |         (her round)          |
```

---

## Kurulum

### Gereksinimler

```bash
pip install torch torchvision matplotlib
```

### Veri Seti

MNIST veri seti `centralized_baseline.py` çalıştırılarak otomatik indirilir, ya da `data/` dizinine manuel olarak yerleştirilebilir.

> **Not:** `federated_learning/utils.py` içindeki `data_path` değişkenini kendi ortamınıza göre güncelleyin.

---

## Kullanım

### 1. Merkezi Eğitim (Baseline)

```bash
cd centralized
python centralized_baseline.py
```

### 2. Federated Learning

**Her istemci için ayrı terminal açın.**

**Sunucuyu başlatın:**

```bash
cd federated_learning
python server.py --num_clients 3 --rounds 10 --port 5000
```

**İstemcileri başlatın** (her biri ayrı terminalde):

```bash
# İstemci 0
python client.py --client_id 0 --num_clients 3 --epochs 2 --server_port 5000

# İstemci 1
python client.py --client_id 1 --num_clients 3 --epochs 2 --server_port 5000

# İstemci 2
python client.py --client_id 2 --num_clients 3 --epochs 2 --server_port 5000
```

#### Parametre Referansı

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|---------|
| `--num_clients` | 3 | İstemci sayısı |
| `--rounds` | 10 | Federated learning round sayısı |
| `--port` | 5000 | Sunucu portu |
| `--client_id` | — | İstemci kimliği (0'dan başlar) |
| `--epochs` | 2 | Yerel eğitim epoch sayısı |
| `--server_host` | localhost | Sunucu adresi |

---

## Deney Sonuçları

### Federated Learning — 10 Round

| İstemci Sayısı | Round 1 Accuracy | Son Accuracy (Round 10) |
|:--------------:|:----------------:|:-----------------------:|
| 3 istemci      | %30.82           | **%93.74**              |
| 5 istemci      | %46.31           | **%89.25**              |
| 10 istemci     | %50.96           | **%89.08**              |

> Merkezi eğitim (20 epoch) karşılaştırma sonuçları için `centralized/centralized_baseline.png` dosyasına bakınız.

### Genel Gözlemler

- **3 istemci** yapılandırması en yüksek doğruluğa (%93.74) ulaşmıştır; çünkü her istemcinin veri payı ve kapsadığı sınıf çeşitliliği görece daha yüksektir.
- **İstemci sayısı arttıkça** Non-IID etkisi belirginleşmekte ve birleşme daha yavaş gerçekleşmektedir.
- Tüm yapılandırmalar 10 round içinde makul doğruluk değerlerine ulaşmıştır.

---

## Non-IID Veri Dağılımı

Her istemciye belirli rakam sınıfları atanarak gerçekçi bir federated senaryo simüle edilmektedir:

```python
# Her istemci num_clients'a göre 2-3 ardışık rakam sınıfı alır
digits_per_client = 2 if num_clients > 5 else 3
```

Bu yaklaşım, gerçek dünya federated learning senaryolarında sıkça karşılaşılan **veri heterojenliğini** yansıtmaktadır.

---

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.
