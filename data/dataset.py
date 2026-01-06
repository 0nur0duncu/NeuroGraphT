"""
Sleep-EDF Dataset Module
Uyku evresi sınıflandırması için veri yükleme ve DataLoader oluşturma
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from .preprocessing import (
    get_subject_files, 
    load_sleep_edf_subject, 
    NUM_CLASSES,
    get_sleep_stage_name
)
from .download import ensure_dataset


class SleepEDFDataset(Dataset):
    """Sleep-EDF Dataset için PyTorch Dataset sınıfı."""
    
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Args:
            signals: Shape (N, 1, T) - epoch'lar
            labels: Shape (N,) - uyku evreleri (0-4)
            transform: Opsiyonel dönüşüm fonksiyonu
        """
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label


def load_sleep_edf_dataset(
    data_dir: str,
    study: str = 'SC',
    channel: str = 'EEG Fpz-Cz',
    max_subjects: Optional[int] = None,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Sleep-EDF veri setini yükle.
    
    Args:
        data_dir: Veri dizini (veya None ise otomatik indirir)
        study: 'SC' (Sleep Cassette) veya 'ST' (Sleep Telemetry)
        channel: EEG kanalı ('EEG Fpz-Cz' veya 'EEG Pz-Oz')
        max_subjects: Maksimum özne sayısı (None = tümü)
        normalize: Z-score normalizasyonu uygula
        verbose: İlerleme göster
    
    Returns:
        signals: Shape (N, 1, T)
        labels: Shape (N,)
        subject_indices: Her epoch'un hangi özneye ait olduğu
    """
    # Veri setini indir (yoksa)
    data_path = ensure_dataset(data_dir, study=study, verbose=verbose)
    
    # Dosya çiftlerini bul
    file_pairs = get_subject_files(str(data_path), study=study)
    
    if len(file_pairs) == 0:
        raise ValueError(f"Veri dosyaları bulunamadı: {data_path}")
    
    if max_subjects is not None:
        file_pairs = file_pairs[:max_subjects]
    
    if verbose:
        print(f"Toplam {len(file_pairs)} özne bulundu")
    
    all_signals = []
    all_labels = []
    subject_indices = []
    
    iterator = tqdm(file_pairs, desc="Özneler yükleniyor") if verbose else file_pairs
    
    for subject_idx, (psg_file, hypno_file) in enumerate(iterator):
        try:
            epochs, labels = load_sleep_edf_subject(
                psg_file, hypno_file, channel=channel, normalize=normalize
            )
            
            if len(epochs) > 0:
                all_signals.append(epochs)
                all_labels.append(labels)
                subject_indices.extend([subject_idx] * len(labels))
        except Exception as e:
            if verbose:
                print(f"Uyarı: {psg_file} yüklenemedi: {e}")
            continue
    
    if len(all_signals) == 0:
        raise ValueError("Hiç veri yüklenemedi!")
    
    signals = np.concatenate(all_signals, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if verbose:
        print(f"\nToplam {len(signals)} epoch yüklendi")
        print("Sınıf dağılımı:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            stage_name = get_sleep_stage_name(label)
            print(f"  {stage_name}: {count} ({100*count/len(labels):.1f}%)")
    
    return signals, labels, subject_indices


def create_data_loaders(
    data_dir: Optional[str] = None,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    study: str = 'SC',
    channel: str = 'EEG Fpz-Cz',
    max_subjects: Optional[int] = None,
    normalize: bool = True,
    random_seed: int = 42,
    num_workers: int = 0,
    verbose: bool = True
) -> Dict[str, DataLoader]:
    """
    Train/Val/Test DataLoader'ları oluştur.
    
    Args:
        data_dir: Veri dizini
        batch_size: Batch boyutu
        train_ratio: Eğitim seti oranı
        val_ratio: Doğrulama seti oranı
        study: SC veya ST
        channel: EEG kanalı
        max_subjects: Maksimum özne sayısı
        normalize: Normalizasyon uygula
        random_seed: Rastgele tohum
        num_workers: DataLoader worker sayısı
        verbose: İlerleme göster
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoader'ları
    """
    # Veri setini yükle
    signals, labels, _ = load_sleep_edf_dataset(
        data_dir=data_dir,
        study=study,
        channel=channel,
        max_subjects=max_subjects,
        normalize=normalize,
        verbose=verbose
    )
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=y_temp
    )
    
    # Dataset'leri oluştur
    train_dataset = SleepEDFDataset(X_train, y_train)
    val_dataset = SleepEDFDataset(X_val, y_val)
    test_dataset = SleepEDFDataset(X_test, y_test)
    
    if verbose:
        print(f"\nVeri seti bölündü:")
        print(f"  Train: {len(train_dataset)} örnek")
        print(f"  Val: {len(val_dataset)} örnek")
        print(f"  Test: {len(test_dataset)} örnek")
    
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }
    
    return loaders


def create_subject_split_loaders(
    data_dir: Optional[str] = None,
    batch_size: int = 64,
    test_subjects: int = 5,
    val_subjects: int = 3,
    study: str = 'SC',
    channel: str = 'EEG Fpz-Cz',
    max_subjects: Optional[int] = None,
    normalize: bool = True,
    random_seed: int = 42,
    num_workers: int = 0,
    verbose: bool = True
) -> Dict[str, DataLoader]:
    """
    Özne bazlı bölme ile DataLoader'ları oluştur.
    (Daha gerçekçi değerlendirme için önerilen yöntem)
    """
    # Veri setini yükle
    signals, labels, subject_indices = load_sleep_edf_dataset(
        data_dir=data_dir,
        study=study,
        channel=channel,
        max_subjects=max_subjects,
        normalize=normalize,
        verbose=verbose
    )
    
    subject_indices = np.array(subject_indices)
    unique_subjects = np.unique(subject_indices)
    n_subjects = len(unique_subjects)
    
    if verbose:
        print(f"\nToplam {n_subjects} özne")
    
    # Özneleri karıştır
    np.random.seed(random_seed)
    np.random.shuffle(unique_subjects)
    
    # Bölme
    test_subj = set(unique_subjects[:test_subjects])
    val_subj = set(unique_subjects[test_subjects:test_subjects + val_subjects])
    train_subj = set(unique_subjects[test_subjects + val_subjects:])
    
    # Maskeleri oluştur
    train_mask = np.isin(subject_indices, list(train_subj))
    val_mask = np.isin(subject_indices, list(val_subj))
    test_mask = np.isin(subject_indices, list(test_subj))
    
    X_train, y_train = signals[train_mask], labels[train_mask]
    X_val, y_val = signals[val_mask], labels[val_mask]
    X_test, y_test = signals[test_mask], labels[test_mask]
    
    # Dataset'leri oluştur
    train_dataset = SleepEDFDataset(X_train, y_train)
    val_dataset = SleepEDFDataset(X_val, y_val)
    test_dataset = SleepEDFDataset(X_test, y_test)
    
    if verbose:
        print(f"Özne bazlı bölme:")
        print(f"  Train: {len(train_subj)} özne, {len(train_dataset)} epoch")
        print(f"  Val: {len(val_subj)} özne, {len(val_dataset)} epoch")
        print(f"  Test: {len(test_subj)} özne, {len(test_dataset)} epoch")
    
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }
    
    return loaders
