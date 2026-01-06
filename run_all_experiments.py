import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Any
import json
from datetime import datetime
from tqdm import tqdm

from models.baselines import BaselineCNNTransformer, Baseline1DCNNTransformer
from models.neurographt import NeuroGraphT
from data.dataset import SleepEDFDataset, load_sleep_edf_dataset
from utils.training import train_one_epoch, validate
from utils.focal_loss import FocalLoss

# Deney konfigürasyonları
EXPERIMENTS = [
    ("CNN-Transformer", None, None),
    ("1D-CNN-Transformer", None, None),
    ("NeuroGraphT", 50, "value"),
    ("NeuroGraphT", 25, "value"),
    ("NeuroGraphT", 50, "connection"),
    ("NeuroGraphT", 25, "connection"),
]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_model(
    model_type: str, 
    num_classes: int, 
    sparsity: float = 50, 
    thresholding: str = "value",
    config: Dict = None
) -> nn.Module:
    """Model oluştur."""
    dropout = config.get("model", {}).get("dropout", 0.1) if config else 0.1
    
    # Transformer parametreleri
    transformer_dim = 128
    num_heads = 8
    num_layers = 4
    
    if config and "model" in config and "transformer" in config["model"]:
        t_config = config["model"]["transformer"]
        transformer_dim = t_config.get("d_model", 128)
        num_heads = t_config.get("num_heads", 8)
        num_layers = t_config.get("num_layers", 4)
    
    if model_type == "CNN-Transformer":
        return BaselineCNNTransformer(
            in_channels=1,
            conv_channels=[32, 64, 128],
            kernel_sizes=[5, 5, 5],
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    
    elif model_type == "1D-CNN-Transformer":
        return Baseline1DCNNTransformer(
            in_channels=1,
            conv_channels=[32, 64, 128, 64],
            kernel_sizes=[7, 5, 5, 3],
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    
    elif model_type.startswith("NeuroGraphT"):
        return NeuroGraphT(
            in_channels=1,
            conv_channels=[32, 64, 128],
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            transformer_layers=num_layers,
            d_ff=512,
            num_nodes=16,
            sparsity=sparsity,
            thresholding=thresholding,
            gcn_hidden=128,
            gcn_layers=3,
            num_classes=num_classes,
            dropout=dropout
        )
    
    raise ValueError(f"Bilinmeyen model türü: {model_type}")


def run_single_experiment(
    model_type: str,
    signals: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    run_id: int,
    sparsity: float = 50,
    thresholding: str = "value",
    model_name: str = "",
    total_runs: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    """Tek bir deney çalıştır."""
    seed = 42 + run_id * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    num_classes = 5  # W, N1, N2, N3, REM
    
    # Veri bölme
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        SleepEDFDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        SleepEDFDataset(X_val, y_val), 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        SleepEDFDataset(X_test, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Model oluştur
    model = create_model(
        model_type, num_classes, sparsity, thresholding, config
    ).to(device)
    
    # Training setup
    training_config = config["training"]
    
    # Loss function selection
    class_weights = None
    if config.get("class_weights", {}).get("enabled", False):
        weights = config["class_weights"].get("weights", [1.0] * num_classes)
        class_weights = weights
        if verbose and run_idx == 0:
            print(f"  ⚖️  Class weights: {weights}")
    
    # Choose loss function
    loss_fn = training_config.get("loss_function", "cross_entropy")
    if loss_fn == "focal":
        gamma = training_config.get("focal_gamma", 2.0)
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        if verbose and run_idx == 0:
            print(f"  🎯 Using Focal Loss (gamma={gamma})")
    else:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        if verbose and run_idx == 0:
            print(f"  📊 Using Cross-Entropy Loss")
    
    optimizer = AdamW(
        model.parameters(), 
        lr=training_config["learning_rate"], 
        weight_decay=training_config["weight_decay"]
    )
    
    num_epochs = training_config["num_epochs"]
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=training_config.get("scheduler", {}).get("min_lr", 1e-6)
    )
    
    # Early stopping
    patience = training_config.get("early_stopping", {}).get("patience", 20)
    min_delta = training_config.get("early_stopping", {}).get("min_delta", 0.001)
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    desc = f"  Run {run_id}/{total_runs}"
    pbar = tqdm(range(1, num_epochs + 1), desc=desc, leave=False, ncols=100) if verbose else range(1, num_epochs + 1)
    
    for epoch in pbar:
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if verbose and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f"{train_metrics['loss']:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.2f}%",
                'best': f"{best_val_acc:.2f}%"
            })
        
        if val_metrics["accuracy"] > best_val_acc + min_delta:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                tqdm.write(f"    Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_acc: {best_val_acc:.2f}%)")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    if verbose:
        tqdm.write(f"    Run {run_id} completed - Test Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1']:.2f}%")
    
    return test_metrics


def run_experiment_suite(
    model_type: str,
    signals: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    num_runs: int = 5,
    sparsity: float = 50,
    thresholding: str = "value",
    model_name: str = ""
) -> Dict[str, float]:
    """Birden fazla run ile deney çalıştır."""
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    for run_id in range(1, num_runs + 1):
        metrics = run_single_experiment(
            model_type=model_type,
            signals=signals,
            labels=labels,
            config=config,
            device=device,
            run_id=run_id,
            sparsity=sparsity,
            thresholding=thresholding,
            model_name=model_name,
            total_runs=num_runs,
            verbose=True
        )
        all_results.append(metrics)
    
    # Ortalama ve standart sapma hesapla
    avg_results = {}
    for key in all_results[0].keys():
        if key != "loss":
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            avg_results[f"{key}_std"] = np.std(values)
    
    # Özet yazdır
    print(f"\n  📊 {model_name} Ortalama Sonuçlar ({num_runs} run):")
    print(f"     Accuracy:  {avg_results['accuracy']:.2f}% ± {avg_results['accuracy_std']:.2f}%")
    print(f"     Precision: {avg_results['precision']:.2f}% ± {avg_results['precision_std']:.2f}%")
    print(f"     Recall:    {avg_results['recall']:.2f}% ± {avg_results['recall_std']:.2f}%")
    print(f"     F1:        {avg_results['f1']:.2f}% ± {avg_results['f1_std']:.2f}%")
    
    return avg_results


def main():
    parser = argparse.ArgumentParser(description="Sleep Stage Classification Experiments")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--max-subjects", type=int, default=None, 
                        help="Maksimum özne sayısı (hızlı test için)")
    parser.add_argument("--channel", type=str, default="EEG Fpz-Cz",
                        choices=["EEG Fpz-Cz", "EEG Pz-Oz"],
                        help="Kullanılacak EEG kanalı")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.max_subjects:
        config["data"]["max_subjects"] = args.max_subjects
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    # Veri setini yükle
    print(f"\n{'#'*70}")
    print(f"# Sleep-EDF Veri Seti Yükleniyor")
    print(f"# Kanal: {args.channel}")
    print(f"{'#'*70}")
    
    signals, labels, _ = load_sleep_edf_dataset(
        data_dir=config["data"].get("data_dir"),
        study=config["data"].get("study", "SC"),
        channel=args.channel,
        max_subjects=config["data"].get("max_subjects"),
        normalize=True,
        verbose=True
    )
    
    print(f"\nVeri yüklendi: {len(signals)} epoch, 5 sınıf (W, N1, N2, N3, REM)")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "num_runs": args.num_runs,
        "channel": args.channel,
        "dataset": "Sleep-EDF",
        "experiments": {}
    }
    
    for model_type, sparsity, thresholding in EXPERIMENTS:
        if model_type == "NeuroGraphT":
            th_str = "DE" if thresholding == "value" else "BE"
            model_name = f"NeuroGraphT_{th_str}(a={sparsity})"
        else:
            model_name = model_type
        
        results = run_experiment_suite(
            model_type=model_type,
            signals=signals,
            labels=labels,
            config=config,
            device=device,
            num_runs=args.num_runs,
            sparsity=sparsity if sparsity else 50,
            thresholding=thresholding if thresholding else "value",
            model_name=model_name
        )
        
        all_results["experiments"][model_name] = results
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Tüm deneyler tamamlandı! Sonuçlar '{args.output}' dosyasına kaydedildi.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
