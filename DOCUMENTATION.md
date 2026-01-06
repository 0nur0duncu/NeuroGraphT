# 📚 NeuroGraphT: EEG Tabanlı Uyku Evrelemesi İçin Çizge Dikkat Ağları ve Transformer Temelli Hibrit Derin Öğrenme Yaklaşımı

**Detaylı Teknik Dokümantasyon**

## 📋 İçindekiler
1. [Proje Özeti](#proje-özeti)
2. [Veri Seti ve Özellikleri](#veri-seti-ve-özellikleri)
3. [Veri Seti: Sleep-EDF (Detaylı)](#veri-seti-sleep-edf-detaylı)
4. [Model Mimarisi: NeuroGraphT](#model-mimarisi-NeuroGraphT)
5. [Veri İşleme Pipeline](#veri-i̇şleme-pipeline)
6. [Eğitim ve Değerlendirme](#eğitim-ve-değerlendirme)
7. [Karşılaştırmalı Deneyler](#karşılaştırmalı-deneyler)

---

## 🎯 Proje Özeti

### Proje Adı
**NeuroGraphT**: EEG Tabanlı Uyku Evrelemesi İçin Çizge Dikkat Ağları ve Transformer Temelli Hibrit Derin Öğrenme Yaklaşımı

### Amaç
EEG (Electroencephalogram) sinyallerinden uyku evrelerini otomatik olarak sınıflandırmak için Graph Neural Network (GNN) ve Transformer tabanlı zamansal kodlayıcı kullanan hibrit bir derin öğrenme mimarisi geliştirmek.

### Yöntem
**GNN ve Transformer Tabanlı Zamansal Kodlama:**

Önerilen **NeuroGraphT** mimarisi üç ana bileşenden oluşur:

1. **Temporal Feature Extraction (CNN-Transformer Encoder)**
   - **1D CNN Layers**: Ham EEG sinyallerinden lokal temporal pattern'ler çıkarır
   - **Transformer Encoder**: Self-attention mekanizması ile uzun-menzilli temporal bağımlılıkları modeller
   - **Avantaj**: LSTM'e göre paralelleştirilebilir, vanishing gradient problemi yok

2. **Adaptive Graph Construction (Graph Builder)**
   - Transformer çıktılarından **dinamik graf yapısı** oluşturur
   - **Node'lar**: Temporal feature'lardan türetilen beyin bölgeleri temsilcileri
   - **Edge'ler**: Pearson korelasyonu ile hesaplanan bölge-bölge etkileşimleri
   - **Sparsity Control**: Value/Connection thresholding ile seyrek graf oluşturma

3. **Graph Neural Network Encoder (GCN)**
   - Multi-layer GCN ile graf-yapısı üzerinde öğrenme
   - **Node feature propagation**: Komşu node'lardan bilgi toplayarak zenginleştirilmiş temsiller
   - **Global pooling**: Tüm node'lardan graf-seviyesi embedding oluşturma

4. **Self-Supervised Pre-training (Özdenetimli Ön Eğitim)**
   - Contrastive learning ile EEG temsillerinin güçlendirilmesi
   - Temporal augmentation ve masking stratejileri
   - Transfer learning ile küçük veri setlerinde performans artışı

### Hedef Problem
**5 Sınıflı Uyku Evresi Sınıflandırması:**
- **W (Wake)**: Uyanıklık - Beta/Gamma aktivitesi dominanttır
- **N1**: Hafif uyku - Theta dalgaları başlar, geçiş evresi
- **N2**: Orta derinlik uyku - K-kompleksleri ve uyku iğcikleri (spindle)
- **N3**: Derin uyku (SWS) - Delta dalgaları dominanttır, slow-wave sleep
- **REM**: REM uykusu - Hızlı göz hareketleri, rüya evresi

---

## 📊 Veri Seti ve Özellikleri

### Veri Seti
**Sleep-EDF Database Expanded** (PhysioNet 1.0.0)
- **Kaynak**: PhysioNet (https://physionet.org/content/sleep-edfx/1.0.0/)
- **Lisans**: Open Database License v1.0
- **Erişim**: Açık kaynak, ücretsiz

### Veri Seti İstatistikleri

| Özellik | Değer |
|---------|-------|
| **Toplam Kayıt** | 197 whole-night polysomnographic (PSG) recordings |
| **Alt Gruplar** | Sleep Cassette (SC): 153 kayıt, Sleep Telemetry (ST): 44 kayıt |
| **Özne Sayısı** | SC: 78 özne (25-101 yaş), ST: 22 özne |
| **Kayıt Türü** | SC: Evde kayıt, ST: Hastane ortamı |
| **Toplam Veri Boyutu** | ~8.1 GB (uncompressed) |

### Sinyal Özellikleri

**EEG Kanalları:**
- **Fpz-Cz** (Frontal-Central): K-kompleks ve spindle detection için optimal
- **Pz-Oz** (Parietal-Occipital): Delta wave detection için optimal
- **Proje Varsayılanı**: Fpz-Cz kanalı

**Teknik Parametreler:**
```yaml
Sampling Rate: 100 Hz (EEG/EOG)
Epoch Duration: 30 saniye (AASM standardı)
Samples per Epoch: 3000 (100 Hz × 30s)
Bit Resolution: 16-bit
Dynamic Range: ±200 µV
Format: European Data Format (EDF/EDF+)
```

### Sınıf Dağılımı (Tipik Sleep Cassette)

| Sınıf | Oran | Açıklama | Zorluk |
|-------|------|----------|--------|
| **W** | ~10-15% | Uyanıklık | Kolay (Beta/Gamma) |
| **N1** | ~5-10% | Hafif uyku | **Zor** (Az örnek, belirsiz) |
| **N2** | ~45-50% | Orta uyku | Orta (Spindle/K-complex) |
| **N3** | ~15-20% | Derin uyku | Kolay (Delta dominanttır) |
| **REM** | ~20-25% | REM uykusu | Orta (Theta + göz hareketi) |

**Dengesiz Veri Problemi:**
- N1 sınıfı **severely underrepresented** (~5-10%)
- N2 sınıfı **overrepresented** (~45-50%)
- **Çözüm**: Weighted Cross-Entropy Loss + Focal Loss + Data Augmentation

### Uyku Evresi Anotasyonları

**Scoring Standardı:** Rechtschaffen & Kales (1968) + AASM adaptasyonu
```python
Sleep Stage Mapping:
  - W (Wake)         → Class 0
  - N1 (Stage 1)     → Class 1
  - N2 (Stage 2)     → Class 2
  - N3 (Stage 3+4)   → Class 3  # AASM'de birleştirilmiş
  - REM (Stage R)    → Class 4
  - Unknown/Movement → -1 (atlanır)
```

### Veri Ön İşleme Pipeline

1. **Otomatik İndirme**: AWS S3 üzerinden boto3 ile PhysioNet bucket'tan
2. **EDF Parsing**: MNE-Python ile ham sinyal yükleme
3. **Hypnogram Alignment**: Annotation-signal senkronizasyonu
4. **Epoch Extraction**: 30s sliding window ile 3000-sample epoch'lar
5. **Normalization**: Subject-wise Z-score normalization
6. **Train/Val/Test Split**: Stratified split (70%/15%/15%) + subject-level ayrımı

### Veri Augmentation Stratejileri

**Temporal Augmentation:**
- **Time Warping**: Temporal distortion (±5%)
- **Jittering**: Gaussian noise injection (SNR: 20-30 dB)
- **Scaling**: Amplitude scaling (0.8-1.2x)
- **Shifting**: Random temporal shift (±0.5s)

**Frequency Domain Augmentation:**
- **Band-pass Filtering**: Random filter shift
- **Frequency Masking**: Belirli frekans bantlarını maskeleme

**Self-Supervised Pre-training için:**
- **Temporal Masking**: Rastgele epoch'ları maskele
- **Contrastive Pairs**: Augmented versions as positive pairs
- **Hard Negative Mining**: Benzer ama farklı sınıftan örnekler

### Veri Kalite Kontrol

**Artifact Detection ve Filtreleme:**
- **Movement Artifacts**: "Movement time" etiketli epoch'lar atlanır
- **Unknown Stages**: "Sleep stage ?" etiketli epoch'lar atlanır
- **Signal Quality Check**: Amplitüd sınırı kontrolü (±200 µV)
- **Continuity Check**: Eksik veya bozuk kanal kontrolü

**Final Dataset Statistics:**
```
Toplam PSG Recordings: 197
Kullanılan Recordings: 153 (SC study)
Ortalama Epoch/Gece: ~800-1200 epoch
Toplam Epoch Sayısı: ~120,000-150,000 epoch (SC)
Train/Val/Test Split: ~105,000 / ~22,500 / ~22,500 epoch
```

### Veri Seti Avantajları

✅ **Büyük ölçekli**: 197 whole-night recordings  
✅ **Çeşitlilik**: Geniş yaş aralığı (25-101)  
✅ **Standardizasyon**: AASM/R&K scoring standardı  
✅ **Açık erişim**: Tekrarlanabilir araştırma  
✅ **Multi-channel**: EEG, EOG, EMG sinyalleri  
✅ **Benchmark**: Literatürde yaygın kullanım  

### Veri Seti Zorlukları

⚠️ **Dengesiz sınıf dağılımı**: N1 sınıfı underrepresented  
⚠️ **Inter-rater variability**: Farklı uzmanlar arası tutarsızlık  
⚠️ **Tek kanal**: Klinik PSG'ye göre sınırlı elektrot  
⚠️ **Artifact'lar**: Evde kayıt nedeniyle hareket artifact'ları  
⚠️ **Subject variability**: Bireysel farklılıklar (yaş, cinsiyet, sağlık)

---

## 📊 Veri Seti: Sleep-EDF (Detaylı)

### Genel Bilgiler

**Sleep-EDF Database** (PhysioNet)
- **Kaynak**: https://physionet.org/content/sleep-edfx/1.0.0/
- **Veri Tipi**: Polisomnografi (PSG) kayıtları
- **Toplam Kayıt**: 197 whole-night PSG recordings
- **Format**: European Data Format (EDF/EDF+)
- **Erişim**: Açık kaynak, ücretsiz

### Veri Seti Yapısı

#### Alt Gruplar
1. **Sleep Cassette (SC)**: 
   - Evde kayıt edilen veriler
   - 153 SC* files / 78 özne
   - 2 gece kayıt (bazı özneler için)
   - Yaş aralığı: 25-101 yaş, sağlıklı Caucasian özneler

2. **Sleep Telemetry (ST)**:
   - Hastanede kayıt edilen veriler
   - 44 ST* files / 22 özne
   - Temazepam etkisi çalışması
   - Daha kontrollü ortam

**Proje varsayılanı: SC (Sleep Cassette)**

### Sinyal Özellikleri

#### EEG Kanalları
Projede 2 ana kanal kullanılabilir:
1. **EEG Fpz-Cz** (Varsayılan)
   - Frontal-Central bölge
   - Uyku iğcikleri ve K-komplekslerini iyi yakalar
   
2. **EEG Pz-Oz**
   - Parietal-Occipital bölge
   - Delta dalgalarını daha iyi gösterir

#### Teknik Parametreler
```yaml
# EEG/EOG Signals
Sampling Rate: 100 Hz
Epoch Duration: 30 saniye
Samples per Epoch: 3000 (100 Hz × 30s)
Bit Resolution: 16-bit
Dynamic Range: ±200 µV (tipik EEG range)

# EMG Signal (SC files)
EMG Sampling: 1 Hz (envelope after rectification)
EMG Unit: µV RMS (root-mean-square)

# Other Signals (SC files)
Respiration: 1 Hz
Body Temperature: 1 Hz
Event Marker: 1 Hz
```

### Veri Ön İşleme

#### 1. **Otomatik İndirme** (`data/download.py`)
```python
from data.download import ensure_dataset

# Veri setini indir (yoksa) veya mevcut olanı kullan
data_path = ensure_dataset(
    data_dir="dataset/sleep-edfx",  # Kaydedilecek dizin
    study='SC',                      # SC veya ST
    force_download=False,            # True ise yeniden indir
    verbose=True                     # İlerleme göster
)
```

**AWS S3'ten İndirme:**
- Boto3 kullanarak PhysioNet S3 bucket'ından indirilir
- Anonim erişim (credentials gerekmez)
- ~8.1 GB total uncompressed (197 PSG recordings)
- Sleep Cassette study: ~6 GB
- Sleep Telemetry study: ~2 GB

#### 2. **Sinyal Yükleme** (`data/preprocessing.py`)
```python
# EDF dosyasından EEG sinyalini yükle
signal, sampling_rate = load_edf_file(
    psg_file="SC4001E0-PSG.edf",
    channel="EEG Fpz-Cz"
)
# signal: (n_samples,) numpy array
# sampling_rate: 100.0 Hz
```

**MNE-Python kullanımı:**
- `mne.io.read_raw_edf()`: Ham EEG verilerini okur
- Otomatik kanal seçimi ve veri tipi dönüşümü

#### 3. **Hypnogram İşleme**
```python
# Uyku evresi anotasyonlarını yükle
hypnogram = load_hypnogram("SC4001E0-Hypnogram.edf")
# [(onset, duration, stage_name), ...]
```

**Sleep Stage Annotations (Rechtschaffen & Kales 1968):**
```python
SLEEP_STAGE_DICT = {
    'Sleep stage W': 0,    # Wake (Uyanıklık)
    'Sleep stage 1': 1,    # N1 (NREM Stage 1)
    'Sleep stage 2': 2,    # N2 (NREM Stage 2)
    'Sleep stage 3': 3,    # N3 (NREM Stage 3 - Deep Sleep)
    'Sleep stage 4': 3,    # N3 (Stage 3 ve 4 modern AASM'de birleştirilir)
    'Sleep stage R': 4,    # REM (Rapid Eye Movement)
    'Sleep stage ?': -1,   # Unknown/Not scored (atlanır)
    'Movement time': -1,   # Movement artifact (atlanır)
}
# Not: Rechtschaffen & Kales manuel (1968) Stage 3 ve 4'ü ayırır,
# ancak modern AASM standardı (2007) bunları N3 olarak birleştirir
```

#### 4. **Epoch Çıkarma**
```python
epochs, labels = extract_epochs(
    signal=signal,           # Ham sinyal
    hypnogram=hypnogram,     # Evre anotasyonları
    sampling_rate=100,       # Hz
    epoch_sec=30            # Her epoch 30 saniye
)
# epochs: (n_epochs, 3000) - Her epoch 3000 örnek
# labels: (n_epochs,) - 0-4 arası sınıf etiketleri
```

#### 5. **Normalizasyon**
```python
def normalize_signal(signal):
    """Z-score normalizasyonu"""
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    return (signal - mean) / std
```

**Neden Z-score?**
- Farklı özneler arası amplitüd farklılıklarını giderir
- Mean=0, Std=1 dağılımı sağlar
- Model eğitimini hızlandırır ve stabilize eder

### Veri Seti İstatistikleri

#### Tipik Sınıf Dağılımı (SC)
```
N2: ~45-50%  (En yaygın uyku evresi)
REM: ~20-25%
N3: ~15-20%
W: ~10-15%
N1: ~5-10%   (En az görülen evre)
```

#### Dengesiz Veri Problemi
**N1 sınıfı az temsil edilir** → Bu nedenle:
- **Weighted Cross-Entropy Loss** kullanımı önerilir
- **F1-Score (Macro)** metriği accuracy'den daha anlamlıdır
- **Stratified Split** ile train/val/test bölme

### PyTorch Dataset Sınıfı

```python
class SleepEDFDataset(Dataset):
    """Sleep-EDF için PyTorch Dataset"""
    
    def __init__(self, signals, labels, transform=None):
        self.signals = torch.FloatTensor(signals)  # (N, 1, 3000)
        self.labels = torch.LongTensor(labels)      # (N,)
        self.transform = transform
    
    def __getitem__(self, idx):
        signal = self.signals[idx]  # (1, 3000)
        label = self.labels[idx]    # scalar
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
```

### Veri Yükleme Örneği

```python
from data import load_sleep_edf_dataset, create_data_loaders

# Basit yükleme
signals, labels, subject_indices = load_sleep_edf_dataset(
    data_dir="dataset/sleep-edfx",
    study='SC',
    channel='EEG Fpz-Cz',
    max_subjects=None,  # Tüm özneler
    normalize=True,
    verbose=True
)

# DataLoader oluşturma
dataloaders = create_data_loaders(
    data_dir="dataset/sleep-edfx",
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,  # test_ratio = 0.15 (otomatik)
    random_seed=42,
    num_workers=0
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

---

## 🏗️ Model Mimarisi: NeuroGraphT

### Genel Bakış

**NeuroGraphT** üç ana bileşenden oluşur:

```
Input EEG Signal (1, 3000)
        ↓
┌───────────────────────────────┐
│   CNN-Transformer Encoder     │ ← Temporal feature extraction
│   - CNN: Local patterns       │
│   - Transformer: Dependencies │
└───────────────────────────────┘
        ↓
    H: (batch, seq_len, hidden)
        ↓
┌───────────────────────────────┐
│      Graph Builder            │ ← Adaptive graph construction
│   - Node creation             │
│   - Adjacency matrix          │
└───────────────────────────────┘
        ↓
    Node Features + Adjacency
        ↓
┌───────────────────────────────┐
│       GCN Encoder             │ ← Graph learning
│   - Graph convolutions        │
│   - Node aggregation          │
└───────────────────────────────┘
        ↓
    Graph Embedding
        ↓
    Classifier (Linear)
        ↓
    Logits (5 classes)
```

---

### 1️⃣ CNN-Transformer Encoder

#### CNN Modülü (`CNNTransformerEncoder`)

**Amaç:** Ham EEG sinyalinden düşük-orta seviye özellikler çıkarmak

```python
# Input: (batch, 1, 3000)
self.cnn = nn.Sequential(
    # Layer 1: 1 → 32 channels
    nn.Conv1d(1, 32, kernel_size=5, padding=2),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.MaxPool1d(2),      # Length: 3000 → 1500
    nn.Dropout(0.1),
    
    # Layer 2: 32 → 64 channels
    nn.Conv1d(32, 64, kernel_size=5, padding=2),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.MaxPool1d(2),      # Length: 1500 → 750
    nn.Dropout(0.1),
    
    # Layer 3: 64 → 128 channels
    nn.Conv1d(64, 128, kernel_size=5, padding=2),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.MaxPool1d(2),      # Length: 750 → 375
    nn.Dropout(0.1),
)

# Adaptive pooling: 375 → 64 (sabit uzunluk)
self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
# Output: (batch, 128, 64)
```

**Özellikler:**
- **Kernel Size 5**: EEG'de tipik 1-50 Hz frekans bantlarını yakalar
- **MaxPool**: Özellik boyutunu azaltır, hesaplama verimliliği
- **BatchNorm**: Eğitimi stabilize eder
- **Dropout**: Overfitting'i önler

#### Transformer Modülü

**Neden LSTM yerine Transformer?**
- ✅ **Paralel hesaplama**: LSTM'den 3-5x hızlı
- ✅ **Long-range dependencies**: Self-attention ile tüm pozisyonlar arasında doğrudan bağlantı
- ✅ **Positional encoding**: Temporal bilgi korunur
- ✅ **Better gradient flow**: Vanishing gradient problemi yok

```python
# CNN çıktısını (batch, 128, 64) → (batch, 64, 128) dönüştür
x = x.permute(0, 2, 1)  # (batch, seq_len=64, channels=128)

# Transformer dim'e projeksiyon
x = self.input_projection(x)  # (batch, 64, 128) → (batch, 64, 128)

# Positional encoding ekle
x = self.positional_encoding(x)

# Transformer encoding
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,          # Model boyutu
    nhead=8,              # Attention head sayısı
    dim_feedforward=512,  # FFN hidden size
    dropout=0.1,
    activation='gelu',    # ReLU yerine GELU (daha smooth)
    batch_first=True      # (batch, seq, feature) formatı
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

H = self.transformer(x)  # (batch, 64, 128)
```

**Positional Encoding:**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Sinüzoidal fonksiyonlar ile pozisyon bilgisi
- Öğrenilmez (fixed), generalization için önemli

**Multi-Head Self-Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- 8 head: Farklı attention pattern'leri öğrenir
- d_k = 128/8 = 16 per head

---

### 2️⃣ Graph Builder (Adaptif Graf Oluşturma)

**Amaç:** Transformer çıktısından beyin bölgeleri arası etkileşim grafı oluşturmak

```python
class GraphBuilder(nn.Module):
    def __init__(
        self,
        num_nodes: int = 16,        # N - Düğüm sayısı
        sparsity: float = 25.0,     # a - Seyreklik (%)
        thresholding: str = "value" # Eşikleme yöntemi
    )
```

#### Adımlar:

**1. Node Feature Çıkarma**
```python
# H: (batch, seq_len=64, hidden=128)
H_flat = H.reshape(batch, -1)  # (batch, 64*128=8192)

# 16 node'a böl
features_per_node = 8192 // 16 = 512
node_features = H_flat.reshape(batch, 16, 512)
# node_features: (batch, 16, 512)
```

**2. Korelasyon Matrisi Hesaplama**
```python
# Z-score normalization
node_norm = (node_features - mean) / std

# Pearson korelasyonu
correlation = torch.bmm(node_norm, node_norm.transpose(-2, -1))
correlation = correlation / feature_dim
# C: (batch, 16, 16) - Her (i,j) düğüm çifti arası korelasyon
```

**3. Adjacency Matrix Oluşturma**

**Yöntem A: Value Thresholding**
```python
def _value_threshold(self, C):
    """
    En yüksek %a korelasyon değerine sahip bağlantıları tut
    """
    percentile = 100 - sparsity  # 100-25 = 75
    threshold = np.percentile(C, percentile)
    adjacency = (C > threshold).float()
    
    # Self-loop'ları kaldır
    adjacency = adjacency * (1 - eye)
    
    return adjacency
```

**Yöntem B: Connection Thresholding**
```python
def _connection_threshold(self, C):
    """
    Her düğüm için en güçlü %a bağlantıyı tut
    """
    n_connections = int(N * sparsity / 100)  # 16 * 0.25 = 4
    
    # Top-k en güçlü bağlantılar
    _, indices = torch.topk(C, k=n_connections, dim=-1)
    
    # Sparse adjacency matrix
    adjacency = torch.zeros_like(C)
    adjacency.scatter_(2, indices, C.gather(2, indices))
    
    # Simetrik yap (undirected graph)
    adjacency = torch.maximum(adjacency, adjacency.transpose(-2, -1))
    
    return adjacency
```

**Sparsity Parametresi (a):**
- **25%**: Daha seyrek graf → Sadece güçlü bağlantılar
- **50%**: Daha yoğun graf → Daha fazla etkileşim

**Thresholding Karşılaştırması:**
| Yöntem | Avantaj | Dezavantaj |
|--------|---------|------------|
| **Value** | Global optimizasyon | Her node farklı sayıda bağlantı |
| **Connection** | Her node'a eşit bağlantı | Bazı zayıf bağlantılar dahil olabilir |

---

### 3️⃣ GCN Encoder (Graph Convolutional Network)

**Amaç:** Graf yapısındaki node feature'ları öğrenerek global graph embedding elde etmek

```python
class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,      # 512 (node feature dim)
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    )
```

#### Graph Convolution İşlemi

```python
def forward(self, x, adj):
    """
    Args:
        x: (batch, N=16, F=512) - Node features
        adj: (batch, N=16, N=16) - Adjacency matrix
    Returns:
        graph_embedding: (batch, hidden_channels=128)
    """
    for layer in self.layers:
        # 1. Komşu node'ların toplam feature'larını hesapla
        neighbor_sum = torch.bmm(adj, x)  # (batch, 16, 512)
        
        # 2. Kendi feature'ı ile birleştir
        combined = x + neighbor_sum        # (batch, 16, 512)
        
        # 3. Linear transformation
        x = layer(combined)                # (batch, 16, 128)
        x = F.relu(x)
        x = self.dropout(x)
    
    # 4. Global pooling (tüm node'ları birleştir)
    graph_embedding = x.max(dim=1)[0]  # (batch, 128)
    
    return graph_embedding
```

**GCN Formülü (Simplified):**
```
h_i^(l+1) = σ(W^(l) · (h_i^(l) + Σ h_j^(l)))
                              j∈N(i)
```
- `h_i^(l)`: Node i'nin l. katmandaki feature'ı
- `N(i)`: Node i'nin komşuları
- `W^(l)`: Öğrenilen weight matrix
- `σ`: Activation function (ReLU)

**Katman Sayısı (num_layers=3):**
- **1 layer**: 1-hop neighbors (direkt komşular)
- **2 layers**: 2-hop neighbors
- **3 layers**: 3-hop neighbors (daha global bilgi)

**Global Max Pooling:**
```python
# Her node'dan en önemli feature'ları al
graph_embedding = x.max(dim=1)[0]
```
- Alternatifler: Mean pooling, Sum pooling
- Max pooling: En diskriminatif feature'ları yakalar

---

### 4️⃣ Classifier (Son Katman)

```python
self.classifier = nn.Linear(gcn_hidden=128, num_classes=5)

logits = self.classifier(graph_embedding)  # (batch, 5)
```

**Output:**
- **Shape:** `(batch, 5)`
- **Logits:** Ham sınıf skorları (softmax öncesi)
- **Classes:** [W, N1, N2, N3, REM]

---

### Tam Pipeline Örneği

```python
# Input
x = torch.randn(8, 1, 3000)  # 8 epoch, 1 kanal, 3000 sample

model = NeuroGraphT(
    in_channels=1,
    conv_channels=[32, 64, 128],
    transformer_dim=128,
    num_heads=8,
    transformer_layers=4,
    d_ff=512,
    num_nodes=16,
    sparsity=25.0,
    thresholding="value",
    gcn_hidden=128,
    gcn_layers=3,
    num_classes=5,
    dropout=0.1
)

# Forward pass
logits = model(x)  # (8, 5)

# Prediction
predictions = logits.argmax(dim=1)  # (8,) - Her epoch için tahmin
```

**Shape Transformations:**
```
(8, 1, 3000) → CNN → (8, 128, 64)
(8, 128, 64) → Transformer → (8, 64, 128)
(8, 64, 128) → GraphBuilder → nodes: (8, 16, 512), adj: (8, 16, 16)
(8, 16, 512) + (8, 16, 16) → GCN → (8, 128)
(8, 128) → Classifier → (8, 5)
```

---

## 🔬 Baseline Modeller

NeuroGraphT'in performansını karşılaştırmak için iki baseline model kullanılır. Her ikisi de **graf modülü içermez**, sadece CNN + Transformer kombinasyonudur.

### 1. BaselineCNNTransformer

**Mimari Özellikler:**
- **CNN Derinliği**: 3 katman
- **Kernel Size**: Sabit (5, 5, 5)
- **Sequence Length**: 64 timestep
- **Pooling Strategy**: Global Average Pooling
- **Graf Modülü**: ❌ YOK

```python
class BaselineCNNTransformer(nn.Module):
    def __init__(
        self,
        conv_channels=[32, 64, 128],     # 3 CNN katmanı
        kernel_sizes=[5, 5, 5],          # Sabit kernel size
        transformer_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=5
    )
```

**Pipeline Akışı:**
```
Input (batch, 1, 3000)
  ↓ CNN [32→64→128] kernel_size=5
  ↓ MaxPool (her katmanda /2)
  ↓ AdaptiveAvgPool1d(64)
  ↓ Shape: (batch, 128, 64)
  ↓
  ↓ Transpose → (batch, 64, 128)
  ↓ Input Projection
  ↓ Positional Encoding
  ↓ Transformer (4 layers, 8 heads)
  ↓ Shape: (batch, 64, 128)
  ↓
  ↓ Global Average Pooling (mean over timesteps)
  ↓ Shape: (batch, 128)
  ↓
  ↓ Classifier (Linear)
Output (batch, 5)
```

**Karakteristik:**
- ✅ **Hızlı eğitim**: Daha az CNN katmanı
- ✅ **Average pooling**: Smooth, dengeli özellik aggregation
- ✅ **64 timestep**: Daha fazla temporal resolution
- ❌ **Sığ CNN**: Daha az feature extraction depth

---

### 2. Baseline1DCNNTransformer

**Mimari Özellikler:**
- **CNN Derinliği**: 4 katman (daha derin!)
- **Kernel Size**: Değişken (7, 5, 5, 3) - Piramit stratejisi
- **Sequence Length**: 32 timestep (daha kompakt)
- **Pooling Strategy**: Global Max Pooling
- **Graf Modülü**: ❌ YOK

```python
class Baseline1DCNNTransformer(nn.Module):
    def __init__(
        self,
        conv_channels=[32, 64, 128, 64],  # 4 CNN katmanı
        kernel_sizes=[7, 5, 5, 3],        # Değişken kernel size
        transformer_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=5
    )
```

**Pipeline Akışı:**
```
Input (batch, 1, 3000)
  ↓ CNN Layer 1: kernel=7 [1→32]   ← Geniş temporal pattern
  ↓ MaxPool → 1500
  ↓ CNN Layer 2: kernel=5 [32→64]
  ↓ MaxPool → 750
  ↓ CNN Layer 3: kernel=5 [64→128]
  ↓ MaxPool → 375
  ↓ CNN Layer 4: kernel=3 [128→64] ← Lokal refinement
  ↓ MaxPool → 187
  ↓ AdaptiveAvgPool1d(32)           ← Daha kısa sequence
  ↓ Shape: (batch, 64, 32)
  ↓
  ↓ Transpose → (batch, 32, 64)
  ↓ Input Projection
  ↓ Positional Encoding
  ↓ Transformer (4 layers, 8 heads)
  ↓ Shape: (batch, 32, 128)
  ↓
  ↓ Global Max Pooling (max over timesteps)  ← Farklı!
  ↓ Shape: (batch, 128)
  ↓
  ↓ Classifier (Linear)
Output (batch, 5)
```

**Karakteristik:**
- ✅ **Derin CNN**: 4 katman, daha zengin feature extraction
- ✅ **Piramit kernel**: 7→5→5→3 (geniş→dar)
- ✅ **Max pooling**: Belirgin özellikleri vurgular
- ✅ **32 timestep**: Daha agresif sıkıştırma, hesaplama tasarrufu
- ❌ **Daha fazla parametre**: Overfitting riski

---

### 🆚 BaselineCNNTransformer vs Baseline1DCNNTransformer

#### Temel Farklar Tablosu

| Özellik | BaselineCNNTransformer | Baseline1DCNNTransformer |
|---------|----------------------|--------------------------|
| **CNN Katman Sayısı** | 3 | 4 (daha derin) |
| **Kernel Size** | [5, 5, 5] (sabit) | [7, 5, 5, 3] (değişken) |
| **İlk Kernel** | 5 | **7** ← Geniş receptive field |
| **Son Kernel** | 5 | **3** ← Lokal refinement |
| **Adaptive Pool** | 64 timestep | 32 timestep (daha kompakt) |
| **Global Pooling** | **Average** | **Max** |
| **Parametre Sayısı** | Daha az | Daha fazla |
| **Eğitim Hızı** | Hızlı | Biraz daha yavaş |
| **Feature Extraction** | Sığ, genel | Derin, detaylı |

#### Kernel Size Stratejisi Karşılaştırması

**BaselineCNNTransformer:**
```
Kernel=5 → Kernel=5 → Kernel=5
  ↓          ↓          ↓
Dengeli    Dengeli    Dengeli
```
- Tüm katmanlarda aynı receptive field
- Uniform feature extraction

**Baseline1DCNNTransformer:**
```
Kernel=7 → Kernel=5 → Kernel=5 → Kernel=3
  ↓          ↓          ↓          ↓
Geniş      Orta       Orta       Dar
Pattern    Features   Features   Details
```
- **İlk katman (7)**: Geniş temporal pattern'ler yakalar (delta dalgaları)
- **Orta katmanlar (5,5)**: Dengeli feature extraction
- **Son katman (3)**: Lokal detayları refine eder (spindle'lar)

#### Pooling Strategy Farkı

**Global Average Pooling (BaselineCNNTransformer):**
```python
x = x.mean(dim=1)  # Tüm timestep'lerin ortalaması
```
- **Smooth aggregation**: Tüm temporal bilgiyi dengeli kullanır
- **Robust**: Outlier'lara duyarlı değil
- **Genel pattern'ler**: Overall aktivite seviyesi
- **Örnek**: N2 evresi → K-kompleks + spindle'ların genel karakteri

**Global Max Pooling (Baseline1DCNNTransformer):**
```python
x = x.max(dim=1)[0]  # Her feature'ın maksimum aktivasyonu
```
- **Discriminative features**: En belirgin özellikleri vurgular
- **Sparse activation**: Kritik anları yakalar
- **Belirgin event'ler**: Spindle zirveleri, K-kompleks amplitudes
- **Örnek**: N2 evresi → En yüksek spindle amplitüdü

#### Sequence Length Etkisi

**64 Timestep (BaselineCNNTransformer):**
- Daha fazla temporal resolution
- Transformer için daha uzun attention
- Daha ince temporal dynamics yakalama
- Hesaplama: O(64²) = 4096 attention operations

**32 Timestep (Baseline1DCNNTransformer):**
- Daha kompakt representation
- Daha hızlı transformer processing
- Agresif feature sıkıştırma
- Hesaplama: O(32²) = 1024 attention operations (4x hızlı!)

#### Hangi Model Ne Zaman İyi?

**BaselineCNNTransformer kullanılmalı:**
- ✅ Smooth, genel pattern'ler önemli olduğunda
- ✅ Overfitting riski yüksek olduğunda (az veri)
- ✅ Hızlı eğitim gerektiğinde
- ✅ Tüm temporal bilgi eşit önemde olduğunda

**Baseline1DCNNTransformer kullanılmalı:**
- ✅ Belirgin, diskriminatif özellikler arandığında
- ✅ Yeterli veri olduğunda (overfitting için)
- ✅ Daha detaylı feature extraction gerektiğinde
- ✅ Kritik event'ler (spindle'lar, K-kompleks) önemli olduğunda

#### Pratik Performans Beklentileri

**Sleep Stage Sınıflandırmada:**

| Sınıf | BaselineCNNTransformer | Baseline1DCNNTransformer |
|-------|----------------------|--------------------------|
| **W (Wake)** | 90% F1 | 91% F1 (max pool iyi) |
| **N1** | 45% F1 | 50% F1 (derin CNN yardımcı) |
| **N2** | 85% F1 | 87% F1 (spindle detection) |
| **N3** | 88% F1 | 89% F1 (delta waves) |
| **REM** | 83% F1 | 84% F1 |
| **Overall** | ~80% | ~82% |

**Gözlem:**
- 1D-CNN-Transformer genellikle +1-2% daha iyi performans
- Ancak daha uzun eğitim süresi gerektirir
- Küçük veri setlerinde overfitting riski

---

### Kod Karşılaştırması

**BaselineCNNTransformer - Forward Pass:**
```python
def forward(self, x):
    # CNN feature extraction
    x = self.cnn(x)                      # (B, 128, L)
    x = self.adaptive_pool(x)            # (B, 128, 64)
    
    # Transformer encoding
    x = x.permute(0, 2, 1)               # (B, 64, 128)
    x = self.input_projection(x)         # (B, 64, 128)
    x = self.positional_encoding(x)
    x = self.transformer(x)              # (B, 64, 128)
    
    # Global average pooling
    x = x.mean(dim=1)                    # (B, 128) ← Average!
    
    # Classification
    x = self.dropout(x)
    return self.fc(x)                    # (B, 5)
```

**Baseline1DCNNTransformer - Forward Pass:**
```python
def forward(self, x):
    # Daha derin CNN feature extraction
    x = self.cnn(x)                      # (B, 64, L) ← 4 layers!
    x = self.adaptive_pool(x)            # (B, 64, 32) ← Shorter!
    
    # Transformer encoding
    x = x.permute(0, 2, 1)               # (B, 32, 64)
    x = self.input_projection(x)         # (B, 32, 128)
    x = self.positional_encoding(x)
    x = self.transformer(x)              # (B, 32, 128)
    
    # Global max pooling
    x = x.max(dim=1)[0]                  # (B, 128) ← Max!
    
    # Classification
    x = self.dropout(x)
    return self.fc(x)                    # (B, 5)
```

---

## � Literatür Karşılaştırması ve Benchmark Sonuçları

### Sleep-EDF Veri Seti Üzerinde State-of-the-Art Sonuçlar

Bu bölüm, **Sleep-EDF Database** üzerinde yapılmış başlıca akademik çalışmaların performans metriklerini içermektedir. Tüm sonuçlar **5-sınıf sınıflandırma** (W, N1, N2, N3, REM) için rapor edilmiştir.

---

### 🏆 Temel Benchmark Modeller

#### 1. **DeepSleepNet** (Supratak et al., 2017)
**Yayın:** IEEE Transactions on Neural Systems and Rehabilitation Engineering  
**Mimari:** CNN-BiLSTM (İki aşamalı öğrenme)

**Sleep-EDF-20 (Fpz-Cz) Sonuçları:**
```
Overall Accuracy: 82.0%
Macro F1-Score:   76.9%
Cohen's Kappa:    0.76

Per-class F1-scores:
  W:   89.2%
  N1:  50.3%  ← En zor sınıf
  N2:  85.1%
  N3:  84.7%
  REM: 81.4%
```

**Mimari Detayları:**
- **Representation Learning**: CNN (small + large filters) + Dropout-based Regularization
- **Sequence Residual Learning**: BiLSTM (2 layers)
- **İki aşamalı eğitim**: Önce CNN, sonra BiLSTM fine-tuning
- **Input**: 30s epoch, tek kanal EEG (Fpz-Cz)
- **Parametre sayısı**: ~3.5M

**Avantajlar:** ✅ Robust temporal modeling, ✅ İki filtre boyutu (3s + 0.5s)  
**Dezavantajlar:** ❌ İki aşamalı eğitim karmaşık, ❌ BiLSTM yavaş

---

#### 2. **U-Time** (Perslev et al., 2019)
**Yayın:** NeurIPS 2019  
**Mimari:** Fully Convolutional U-Net (Temporal segmentation)

**Sleep-EDF-153 Sonuçları:**
```
Overall Accuracy: 81.7% (mean)
Cohen's Kappa:    0.75 ± 0.08
Macro F1-Score:   ~75%

Hiperparametre robustluğu: Çok yüksek
Cross-dataset transfer: Mükemmel
```

**Mimari Detayları:**
- **U-Net Encoder-Decoder**: 12-layer fully convolutional
- **Segment-to-segment mapping**: Her zaman adımı için sınıf tahmini
- **Multi-resolution feature extraction**: Skip connections
- **No recurrence**: Tamamen feed-forward
- **Input**: Uzun sequence (>30s), flexible length

**Avantajlar:** ✅ Hiperparametre robustluğu, ✅ Transfer learning için mükemmel, ✅ Hızlı  
**Dezavantajlar:** ❌ Temporal context sınırlı, ❌ LSTM kadar sequential modeling yok

---

#### 3. **L-SeqSleepNet** (Phan et al., 2023)
**Yayın:** IEEE Journal of Biomedical and Health Informatics  
**Mimari:** Long-sequence modeling with Hierarchical RNN

**Sleep-EDF-20 (Fpz-Cz) Sonuçları:**
```
Overall Accuracy: 83.4%
Macro F1-Score:   78.2%
Cohen's Kappa:    0.78

Per-class F1-scores:
  W:   90.1%
  N1:  53.8%  ← İyileşme!
  N2:  86.5%
  N3:  85.2%
  REM: 82.5%
```

**Mimari Detayları:**
- **Whole-cycle modeling**: ~90 dakikalık sequence (180 epoch)
- **Hierarchical architecture**: Epoch-level → Cycle-level
- **Adaptive Feature Recalibration**: Attention-like mechanism
- **Training strategy**: Curriculum learning

**Avantajlar:** ✅ Uzun-menzilli temporal bağımlılıklar, ✅ N1 sınıfında iyileşme  
**Dezavantajlar:** ❌ Çok uzun sequence gerektirir, ❌ Hafıza yoğun

---

#### 4. **NeuroNet** (Lee et al., 2024)
**Yayın:** arXiv 2404.17585 (Under review)  
**Mimari:** Self-supervised pre-training + CNN-Transformer

**Sleep-EDF-20 Sonuçları:**
```
Overall Accuracy: 84.7%  ← SOTA!
Macro F1-Score:   80.5%
Cohen's Kappa:    0.80

Self-supervised pre-training benefits:
  - Baseline (from scratch): 82.0% Acc
  - With pre-training:       84.7% Acc (+2.7%)
```

**Mimari Detayları:**
- **Pre-training**: Contrastive learning + Temporal masking
- **Architecture**: 1D CNN + Multi-head self-attention
- **Data augmentation**: Time warping, jittering, masking
- **Fine-tuning**: Task-specific classifier

**Avantajlar:** ✅ Self-supervised learning, ✅ Az veriyle yüksek performans  
**Dezavantajlar:** ❌ Pre-training maliyeti yüksek

---

#### 5. **SleepTransformer** (Phan et al., 2022)
**Yayın:** arXiv 2211.13005  
**Mimari:** Pure Transformer (CNN-free)

**Sleep-EDF Sonuçları:**
```
Overall Accuracy: 83.1%
Macro F1-Score:   77.8%
Cohen's Kappa:    0.77

Inference speed: 3x faster than BiLSTM
```

**Mimari Detayları:**
- **Multi-scale Transformer**: Farklı temporal resolution'larda attention
- **Positional encoding**: Learnable + Sinusoidal
- **No CNN**: Doğrudan EEG raw signal üzerinde
- **Efficiency**: Linear attention (Linformer-style)

**Avantajlar:** ✅ Hızlı inference, ✅ Paralel eğitim  
**Dezavantajlar:** ❌ CNN'siz lokal pattern yakalama zor

---

### 📊 Performans Karşılaştırma Tablosu

| Model | Year | Architecture | Accuracy | F1-Macro | Kappa | N1 F1 | Params |
|-------|------|-------------|----------|----------|-------|-------|--------|
| **DeepSleepNet** | 2017 | CNN-BiLSTM | 82.0% | 76.9% | 0.76 | 50.3% | 3.5M |
| **U-Time** | 2019 | U-Net FCN | 81.7% | ~75% | 0.75 | ~48% | 2.8M |
| **SleepTransformer** | 2022 | Pure Transformer | 83.1% | 77.8% | 0.77 | 52.1% | 4.2M |
| **L-SeqSleepNet** | 2023 | Hierarchical RNN | 83.4% | 78.2% | 0.78 | 53.8% | 5.1M |
| **NeuroNet (SSL)** | 2024 | CNN-Transformer + SSL | **84.7%** | **80.5%** | **0.80** | **55.2%** | 3.8M |
| **NeuroGraphT (Ours)** | 2026 | CNN-Transformer-GNN | **🎯 Target** | **🎯 Target** | **🎯 Target** | **🎯 Target** | ~4.5M |

**Notlar:**
- Tüm sonuçlar Sleep-EDF-20 (Fpz-Cz) üzerinde 5-class classification
- N1 F1: En zor sınıf, literatürde genellikle 45-55% aralığında
- Kappa: Cohen's Kappa coefficient (inter-rater agreement metric)

---

### 🔬 Graf Tabanlı Yaklaşımlar (Yeni Trend!)

#### **GraphSleepNet** (Jia et al., 2020)
**Yayın:** EMBC 2020  
**Mimari:** GCN + LSTM

**Multi-channel EEG (6 kanal) Sonuçları:**
```
Overall Accuracy: 85.2%
Macro F1-Score:   80.8%
```

**Not:** Multi-channel kullanıyor (6 EEG + 2 EOG), tek kanalla karşılaştırma zor!

#### **Spatial-Temporal GNN** (Shi et al., 2021)
**Mimari:** Spatial GCN + Temporal GCN

**Avantaj:** Electrode-level graph + Temporal graph  
**Dezavantaj:** Pre-defined electrode graph (fixed topology)

---

### 🎯 NeuroGraphT'nin Yenilikçi Katkıları

| Özellik | DeepSleepNet | U-Time | L-SeqSleepNet | NeuroNet | **NeuroGraphT (Ours)** |
|---------|--------------|--------|---------------|----------|----------------------|
| **Architecture** | CNN-BiLSTM | U-Net FCN | Hierarchical RNN | CNN-Transformer | **CNN-Transformer-GCN** |
| **Temporal Modeling** | BiLSTM | U-Net Conv | BiLSTM | Transformer | ✅ **Transformer** |
| **Graph Learning** | ❌ | ❌ | ❌ | ❌ | ✅ **Adaptive GCN** |
| **Self-Supervised** | ❌ | ❌ | ❌ | ✅ | ✅ **Contrastive + Masking** |
| **Dynamic Graph** | ❌ | ❌ | ❌ | ❌ | ✅ **Data-driven adjacency** |
| **Single-channel** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Training Speed** | Slow | Fast | Very Slow | Medium | Medium |

**Bizim Yeniliklerimiz:**
1. ✅ **Adaptive Graph Construction**: Data-driven adjacency matrix (korelasyon tabanlı)
2. ✅ **Hybrid Architecture**: CNN + Transformer + GCN (üç modül birlikte)
3. ✅ **Sparsity Control**: Value/Connection thresholding ile seyrek graf
4. ✅ **Self-Supervised Pre-training**: EEG temsillerini güçlendirme
5. ✅ **Transformer for Temporal**: LSTM'den hızlı ve robust

---

### 📈 Beklenen Hedef Performans

**Sleep-EDF-153 (Tüm SC dataset) için:**

```yaml
Target Metrics:
  Overall Accuracy:  84-86%  # NeuroNet'i geçmek
  Macro F1-Score:    80-82%  # State-of-the-art seviye
  Cohen's Kappa:     0.78-0.82
  
Per-class F1 Targets:
  W:    90-92%   # Kolay sınıf
  N1:   54-58%   # Kritik sınıf - literatürden iyi
  N2:   86-88%   # En yaygın sınıf
  N3:   85-88%   # Delta detection
  REM:  83-85%   # REM detection
```

**Graf Modülünün Beklenen Katkısı:**
- Baseline (CNN-Transformer only): ~82-83% accuracy
- **+Graf Modülü**: ~84-86% accuracy (**+2-3%** artış bekleniyor)
- **+Self-supervised pre-training**: ~+1-2% ek boost

---

### 🔍 Literatür Analizi: Kritik Gözlemler

**1. N1 Sınıfı Zorlukları:**
- Tüm modeller N1'de düşük performans (~45-55% F1)
- Neden: Az örnek + belirsiz özellikler + N2'ye geçiş evresi
- Çözüm: Weighted loss + data augmentation + temporal context

**2. Temporal Modeling Trendi:**
- 2017-2020: LSTM/BiLSTM dominanttı
- 2021-2024: Transformer'a geçiş (**3-5x hızlı**)
- 2024+: Self-attention + Graph learning kombinasyonu

**3. Self-Supervised Learning Impact:**
- NeuroNet (2024): +2.7% accuracy boost with SSL
- Trend: Pre-training stratejileri popülaritesi artıyor
- Küçük veri setlerinde kritik önemi var

**4. Graf Tabanlı Yaklaşımların Potansiyeli:**
- Multi-channel EEG'de başarılı (85%+ accuracy)
- Tek kanal için henüz yeterli çalışma yok
- Adaptive graph construction: Yeni araştırma alanı

---

### 📖 Referans Makaleler

1. **Supratak et al. (2017)** - DeepSleepNet  
   IEEE Trans. Neural Syst. Rehabil. Eng. | Citations: 1200+

2. **Perslev et al. (2019)** - U-Time  
   NeurIPS 2019 | Citations: 400+

3. **Phan et al. (2023)** - L-SeqSleepNet  
   IEEE J. Biomed. Health Inform. | Citations: 80+

4. **Lee et al. (2024)** - NeuroNet (Self-supervised)  
   arXiv:2404.17585 | Under Review

5. **Jia et al. (2020)** - GraphSleepNet  
   EMBC 2020 | Citations: 120+

---

## �📈 Eğitim ve Değerlendirme

### Loss Function

```python
# Weighted Cross-Entropy (dengesiz veri için)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(class_weights)
)
```

**Neden Weighted?**
- N1 sınıfı çok az (~5%) → Ağırlık artırılır
- N2 sınıfı çok fazla (~45%) → Ağırlık azaltılır

### Optimizer

```python
optimizer = AdamW(
    model.parameters(),
    lr=0.0001,           # Transformer için düşük LR
    weight_decay=0.01    # L2 regularization
)
```

**AdamW > Adam:**
- Decoupled weight decay
- Daha iyi generalization

### Learning Rate Scheduler

```python
# Warmup + Cosine Annealing
warmup = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=5  # 5 epoch warmup
)

cosine = CosineAnnealingLR(
    optimizer,
    T_max=100,     # Max epoch
    eta_min=1e-6   # Minimum LR
)

scheduler = SequentialLR(
    optimizer,
    [warmup, cosine],
    milestones=[5]
)
```

### Eğitim Döngüsü

```python
for epoch in range(num_epochs):
    # Training
    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validation
    val_metrics = validate(
        model, val_loader, criterion, device
    )
    
    # LR scheduling
    scheduler.step()
    
    # Checkpoint saving
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        torch.save(model.state_dict(), 'best_model.pt')
```

### Değerlendirme Metrikleri

```python
from utils.metrics import calculate_all_metrics

metrics = calculate_all_metrics(y_true, y_pred)
# {
#     'accuracy': 85.3,     # Genel doğruluk
#     'precision': 82.1,    # Macro-average precision
#     'recall': 81.7,       # Macro-average recall
#     'f1': 81.9            # Macro-average F1 (en önemli)
# }
```

**Neden Macro F1?**
- Her sınıfı eşit önemde değerlendirir
- Dengesiz veri setlerinde daha güvenilir
- N1 gibi az temsil edilen sınıfları da dikkate alır

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['W', 'N1', 'N2', 'N3', 'REM'],
            yticklabels=['W', 'N1', 'N2', 'N3', 'REM'])
```

**Yorumlama:**
- Diagonal: Doğru tahminler
- Off-diagonal: Karışıklıklar (ör. N1 vs N2)

---

## 🔬 Karşılaştırmalı Deneyler

### Deney Konfigürasyonları

```python
EXPERIMENTS = [
    # Baseline modeller (graf yok)
    ("CNN-Transformer", None, None),
    ("1D-CNN-Transformer", None, None),
    
    # NeuroGraphT varyasyonları
    ("NeuroGraphT", 50, "value"),
    ("NeuroGraphT", 25, "value"),
    ("NeuroGraphT", 50, "connection"),
    ("NeuroGraphT", 25, "connection"),
]
```

### Parametre Analizi

| Model | Sparsity | Threshold | Node Count | Açıklama |
|-------|----------|-----------|------------|----------|
| Baseline | - | - | - | Graf yok |
| NeuroGraphT-V50 | 50% | Value | 16 | Yoğun graf |
| NeuroGraphT-V25 | 25% | Value | 16 | Seyrek graf |
| NeuroGraphT-C50 | 50% | Connection | 16 | Her node'da 8 bağlantı |
| NeuroGraphT-C25 | 25% | Connection | 16 | Her node'da 4 bağlantı |

### Çalıştırma

```bash
# Tüm deneyleri çalıştır
python run_all_experiments.py \
    --num-runs 3 \
    --max-subjects 10 \
    --config config/config.yaml

# K-fold cross validation
python run_experiments_kfold.py \
    --k-folds 5 \
    --config config/config.yaml
```

### Sonuç Analizi

```python
import json

with open("results.json") as f:
    results = json.load(f)

for model_name, metrics in results["experiments"].items():
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}% ± {metrics['accuracy_std']:.2f}%")
    print(f"  F1 Score: {metrics['f1']:.2f}% ± {metrics['f1_std']:.2f}%")
```

### Beklenen Sonuçlar (Literatür)

| Model Type | Accuracy | F1-Score | Notlar |
|------------|----------|----------|--------|
| CNN-LSTM | ~78-82% | ~75-79% | Baseline |
| CNN-Transformer | ~80-84% | ~77-81% | LSTM'den iyi |
| **NeuroGraphT** | ~82-86% | ~79-83% | Graf ile artış |

**Graf Modülünün Katkısı:**
- ✅ +2-4% accuracy
- ✅ +2-3% F1-score
- ✅ Özellikle N1 ve N3 sınıflarında iyileşme

---

## 🚀 Kullanım Örnekleri

### Hızlı Başlangıç (Jupyter Notebook)

```python
# 1. Veri setini indir
from data.download import ensure_dataset
data_path = ensure_dataset(verbose=True)

# 2. Model oluştur
from models import NeuroGraphT
model = NeuroGraphT(num_classes=5)

# 3. Basit test
import torch
test_input = torch.randn(2, 1, 3000)
output = model(test_input)
print(output.shape)  # (2, 5)
```

### Python Script ile Eğitim

```python
from data import create_data_loaders
from models import NeuroGraphT
import torch
import torch.nn as nn

# DataLoader
loaders = create_data_loaders(
    batch_size=32,
    max_subjects=20,
    verbose=True
)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuroGraphT(
    num_classes=5,
    sparsity=25.0,
    thresholding="value"
).to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

for epoch in range(100):
    # Training loop...
    pass
```

---

## 📊 Hiperparametre Tablosu

### CNN Parametreleri
```yaml
in_channels: 1
conv_channels: [32, 64, 128]
kernel_sizes: [5, 5, 5]
pool_size: 2
```

### Transformer Parametreleri
```yaml
d_model: 128
num_heads: 8
num_layers: 4
d_ff: 512
dropout: 0.1
```

### Graf Parametreleri
```yaml
num_nodes: 16
sparsity: 25  # veya 50
thresholding: "value"  # veya "connection"
```

### GCN Parametreleri
```yaml
hidden_channels: 128
num_layers: 3
dropout: 0.1
```

### Eğitim Parametreleri
```yaml
batch_size: 32
num_epochs: 100
learning_rate: 0.0001
weight_decay: 0.01
warmup_epochs: 5
```

---

## 🎓 Temel Kavramlar

### EEG (Electroencephalogram)
Beyin elektriksel aktivitesinin kayıt edilmesi. Uyku evrelerinde farklı frekans bantları dominanttır:
- **Delta (<4 Hz)**: Derin uyku (N3, slow-wave sleep)
- **Theta (4-7 Hz)**: Hafif uyku (N1, drowsiness)
- **Alpha (8-12 Hz)**: Uyanıklık (gözler kapalı, relaxed wakefulness)
- **Beta (13-30 Hz)**: Aktif uyanıklık (alert, active thinking)
- **Gamma (>30 Hz, typically ~30-100 Hz)**: Yüksek konsantrasyon, cross-modal sensory processing

### Transformer
Self-attention mekanizması kullanan model. Her pozisyon, tüm diğer pozisyonlara attention yapabilir.

### Graph Neural Network
Graf yapılı veriler üzerinde çalışan sinir ağları. Node'lar arası ilişkileri öğrenir.

### Sparsity (Seyreklik)
Graf yapısındaki bağlantı oranı. Düşük sparsity = daha az bağlantı = daha seyrek graf.

---

## 📚 Referanslar

### Veri Seti
- **Sleep-EDF Database**: https://physionet.org/content/sleep-edfx/1.0.0/
- Kemp, B., et al. (2000). "Analysis of a sleep-dependent neuronal feedback loop"

### Metodoloji
- **Transformer**: Vaswani et al. (2017) "Attention Is All You Need"
- **GCN**: Kipf & Welling (2017) "Semi-Supervised Classification with Graph Convolutional Networks"
- **Sleep Stage Classification**: Phan et al. (2019) "DeepSleepNet"

---

## 🛠️ Geliştirme Notları

### Performans Optimizasyonları
1. **Mixed Precision Training**: `torch.cuda.amp` ile 2x hızlanma
2. **DataLoader Workers**: `num_workers=4` ile veri yükleme paralelleştirme
3. **Gradient Accumulation**: Daha büyük batch size simulasyonu

### Gelecek İyileştirmeler
- [ ] Multi-channel EEG desteği (Fpz-Cz + Pz-Oz)
- [ ] Attention visualization
- [ ] Graph structure analysis
- [ ] Real-time inference optimization
- [ ] Transfer learning (farklı veri setleri)

---

## 📞 İletişim ve Destek

**Proje Sahibi:** [GitHub Repository]

**Lisans:** MIT

**Son Güncelleme:** 6 Ocak 2025

---

**Not:** Bu dokümantasyon, projenin teknik detaylarını ve kullanımını kapsamlı şekilde açıklamaktadır. Sorularınız için issue açabilir veya katkıda bulunabilirsiniz.
