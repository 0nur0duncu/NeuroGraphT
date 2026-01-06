"""
Sleep-EDF Veri Seti Preprocessing Modülü
Fpz-Cz veya Pz-Oz kanalı kullanarak uyku evresi sınıflandırması
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import re

try:
    import mne
except ImportError:
    print("mne kuruluyor...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mne", "-q"])
    import mne


# Uyku evresi etiketleri (AASM standardına göre)
# W=Wake, N1=Stage 1, N2=Stage 2, N3=Stage 3/4 (SWS), REM=REM
SLEEP_STAGE_DICT = {
    'Sleep stage W': 0,      # Wake
    'Sleep stage 1': 1,      # N1
    'Sleep stage 2': 2,      # N2
    'Sleep stage 3': 3,      # N3 (SWS)
    'Sleep stage 4': 3,      # N3 (SWS) - Stage 3 ve 4 birleştirilir
    'Sleep stage R': 4,      # REM
    'Sleep stage ?': -1,     # Unknown (atlanacak)
    'Movement time': -1,     # Movement (atlanacak)
}

# 5 sınıf: W, N1, N2, N3, REM
NUM_CLASSES = 5
EPOCH_SEC = 30  # Her epoch 30 saniye


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalizasyonu uygula."""
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (signal - mean) / std


def add_channel_dimension(signal: np.ndarray) -> np.ndarray:
    """Kanal boyutu ekle (CNN için)."""
    if signal.ndim == 1:
        return signal.reshape(1, -1)
    elif signal.ndim == 2:
        return signal[:, np.newaxis, :]
    return signal


def preprocess_eeg(
    signal: np.ndarray,
    normalize: bool = True,
    add_channel: bool = True
) -> np.ndarray:
    """EEG sinyalini ön işle."""
    processed = signal.copy()
    
    if normalize:
        processed = normalize_signal(processed)
    
    if add_channel:
        processed = add_channel_dimension(processed)
    
    return processed


def load_edf_file(
    psg_file: str,
    channel: str = 'EEG Fpz-Cz'
) -> Tuple[np.ndarray, float]:
    """
    EDF dosyasından EEG sinyalini yükle.
    
    Args:
        psg_file: PSG EDF dosyasının yolu
        channel: Kullanılacak EEG kanalı ('EEG Fpz-Cz' veya 'EEG Pz-Oz')
    
    Returns:
        signal: EEG sinyali
        sampling_rate: Örnekleme frekansı
    """
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    
    # Kanal isimlerini kontrol et
    available_channels = raw.ch_names
    
    # Kanal adı eşleştirme
    channel_mapping = {
        'EEG Fpz-Cz': ['EEG Fpz-Cz', 'EEG FpzCz', 'EEG Fpz Cz'],
        'EEG Pz-Oz': ['EEG Pz-Oz', 'EEG PzOz', 'EEG Pz Oz']
    }
    
    selected_channel = None
    for ch_variant in channel_mapping.get(channel, [channel]):
        if ch_variant in available_channels:
            selected_channel = ch_variant
            break
    
    if selected_channel is None:
        # Herhangi bir EEG kanalı bul
        for ch in available_channels:
            if 'EEG' in ch:
                selected_channel = ch
                break
    
    if selected_channel is None:
        raise ValueError(f"EEG kanalı bulunamadı. Mevcut kanallar: {available_channels}")
    
    # Kanalı seç ve veriyi al
    raw.pick_channels([selected_channel])
    signal = raw.get_data()[0]
    sampling_rate = raw.info['sfreq']
    
    return signal, sampling_rate


def load_hypnogram(hypno_file: str) -> List[Tuple[float, float, str]]:
    """
    Hypnogram EDF+ dosyasından uyku evrelerini yükle.
    
    Returns:
        List of (onset, duration, stage_name) tuples
    """
    annotations = mne.read_annotations(hypno_file)
    
    stages = []
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        stages.append((onset, duration, description))
    
    return stages


def extract_epochs(
    signal: np.ndarray,
    hypnogram: List[Tuple[float, float, str]],
    sampling_rate: float,
    epoch_sec: int = EPOCH_SEC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sinyali epoch'lara böl ve etiketleri çıkar.
    
    Args:
        signal: Ham EEG sinyali
        hypnogram: Uyku evreleri listesi
        sampling_rate: Örnekleme frekansı (Hz)
        epoch_sec: Her epoch'un süresi (saniye)
    
    Returns:
        epochs: Shape (n_epochs, samples_per_epoch)
        labels: Shape (n_epochs,)
    """
    samples_per_epoch = int(epoch_sec * sampling_rate)
    
    epochs = []
    labels = []
    
    for onset, duration, stage_name in hypnogram:
        # Evresi geçersizse atla
        if stage_name not in SLEEP_STAGE_DICT:
            continue
        
        label = SLEEP_STAGE_DICT[stage_name]
        if label == -1:  # Unknown veya Movement
            continue
        
        # Bu anotasyondaki epoch sayısı
        n_epochs_in_annotation = int(duration // epoch_sec)
        
        for i in range(n_epochs_in_annotation):
            start_sample = int((onset + i * epoch_sec) * sampling_rate)
            end_sample = start_sample + samples_per_epoch
            
            # Sinyal sınırlarını kontrol et
            if end_sample > len(signal):
                break
            
            epoch_data = signal[start_sample:end_sample]
            
            # Epoch uzunluğunu kontrol et
            if len(epoch_data) == samples_per_epoch:
                epochs.append(epoch_data)
                labels.append(label)
    
    return np.array(epochs, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_subject_files(
    data_dir: str,
    study: str = 'SC'
) -> List[Tuple[str, str]]:
    """
    Veri dizininden PSG ve Hypnogram dosya çiftlerini bul.
    
    Args:
        data_dir: Veri dizini
        study: 'SC' (Sleep Cassette) veya 'ST' (Sleep Telemetry)
    
    Returns:
        List of (psg_file, hypno_file) tuples
    """
    data_path = Path(data_dir)
    
    # Olası dizinler
    study_dirs = {
        'SC': ['sleep-cassette', 'SC', ''],
        'ST': ['sleep-telemetry', 'ST', '']
    }
    
    search_dirs = []
    for subdir in study_dirs.get(study, ['']):
        possible_dir = data_path / subdir if subdir else data_path
        if possible_dir.exists():
            search_dirs.append(possible_dir)
    
    file_pairs = []
    prefix = 'SC' if study == 'SC' else 'ST'
    
    for search_dir in search_dirs:
        # PSG dosyalarını bul
        psg_pattern = f"{prefix}*PSG.edf"
        psg_files = list(search_dir.glob(psg_pattern))
        
        for psg_file in psg_files:
            # Eşleşen hypnogram dosyasını bul
            # SC4001E0-PSG.edf -> SC4001EC-Hypnogram.edf (veya EH, veya EP)
            base_name = psg_file.stem.replace('-PSG', '')
            
            # Hypnogram dosya kalıpları
            hypno_patterns = [
                f"{base_name}*Hypnogram.edf",
                f"{base_name.replace('E0', 'E')}*Hypnogram.edf",
            ]
            
            hypno_file = None
            for pattern in hypno_patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    hypno_file = matches[0]
                    break
            
            if hypno_file is not None and hypno_file.exists():
                file_pairs.append((str(psg_file), str(hypno_file)))
    
    return sorted(file_pairs)


def load_sleep_edf_subject(
    psg_file: str,
    hypno_file: str,
    channel: str = 'EEG Fpz-Cz',
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tek bir öznenin verilerini yükle.
    
    Returns:
        epochs: Shape (n_epochs, 1, samples_per_epoch) - kanal boyutu ekli
        labels: Shape (n_epochs,)
    """
    # EEG sinyalini yükle
    signal, sampling_rate = load_edf_file(psg_file, channel)
    
    # Hypnogram'ı yükle
    hypnogram = load_hypnogram(hypno_file)
    
    # Epoch'ları çıkar
    epochs, labels = extract_epochs(signal, hypnogram, sampling_rate)
    
    if len(epochs) == 0:
        return np.array([]), np.array([])
    
    # Normalize et
    if normalize:
        epochs = normalize_signal(epochs)
    
    # Kanal boyutu ekle (N, 1, T)
    epochs = epochs[:, np.newaxis, :]
    
    return epochs, labels


def get_sleep_stage_name(label: int) -> str:
    """Etiket numarasından uyku evresi adını döndür."""
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    return stage_names.get(label, 'Unknown')
