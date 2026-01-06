import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, List
import json
from datetime import datetime
from tqdm import tqdm

from models.baselines import BaselineCNNTransformer, Baseline1DCNNTransformer
from models.neurographt import NeuroGraphT
from data.dataset import SleepEDFDataset, load_sleep_edf_dataset
from utils.training import train_one_epoch, validate
from utils.focal_loss import FocalLoss

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


def run_single_fold(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    fold_id: int,
    sparsity: float = 50,
    thresholding: str = "value",
    num_classes: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    """Tek bir fold çalıştır."""
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
    
    model = create_model(
        model_type, num_classes, sparsity, thresholding, config
    ).to(device)
    
    training_config = config["training"]
    
    # Loss function selection
    class_weights = None
    if config.get("class_weights", {}).get("enabled", False):
        weights = config["class_weights"].get("weights", [1.0] * num_classes)
        class_weights = weights  # Keep as list for FocalLoss
        if verbose and fold_id == 1:
            print(f"  ⚖️  Class weights: {weights}")
    
    # Choose loss function
    loss_fn = training_config.get("loss_function", "cross_entropy")
    if loss_fn == "focal":
        gamma = training_config.get("focal_gamma", 2.0)
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        if verbose and fold_id == 1:
            print(f"  🎯 Using Focal Loss (gamma={gamma})")
    else:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        if verbose and fold_id == 1:
            print(f"  📊 Using Cross-Entropy Loss")
    
    optimizer = AdamW(
        model.parameters(), 
        lr=training_config["learning_rate"], 
        weight_decay=training_config["weight_decay"]
    )
    
    num_epochs = training_config["num_epochs"]
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=training_config.get("scheduler", {}).get("min_lr", 1e-6)
    )
    
    patience = 20
    min_delta = 0.001
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    pbar = tqdm(range(1, num_epochs + 1), desc=f"  Fold {fold_id}", leave=False, ncols=100) if verbose else range(1, num_epochs + 1)
    
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
                tqdm.write(f"    Early stopping at epoch {epoch} (best: {best_epoch})")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test with per-class F1 scores
    class_names = ["W", "N1", "N2", "N3", "REM"]
    test_metrics = validate(
        model, test_loader, criterion, device, 
        class_names=class_names, 
        include_per_class=config.get("evaluation", {}).get("per_class_metrics", False)
    )
    
    if verbose:
        tqdm.write(f"    Fold {fold_id} - Test Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1']:.2f}%")
        if "per_class_f1" in test_metrics:
            per_class_f1 = test_metrics["per_class_f1"]
            tqdm.write(f"      Per-class F1: W={per_class_f1['W']:.1f}%, N1={per_class_f1['N1']:.1f}%, N2={per_class_f1['N2']:.1f}%, N3={per_class_f1['N3']:.1f}%, REM={per_class_f1['REM']:.1f}%")
    
    return test_metrics


def run_kfold_experiment(
    model_type: str,
    signals: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    n_splits: int = 10,
    sparsity: float = 50,
    thresholding: str = "value",
    model_name: str = "",
    seed: int = 42
) -> Dict[str, float]:
    """K-Fold cross validation deneyi."""
    num_classes = 5  # W, N1, N2, N3, REM
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({n_splits}-Fold CV)")
    print(f"{'='*60}")
    
    # Outer K-fold for test
    outer_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    all_results = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_kfold.split(signals, labels), 1):
        X_train_val, X_test = signals[train_val_idx], signals[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]
        
        inner_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed + fold_idx)
        train_idx, val_idx = next(inner_kfold.split(X_train_val, y_train_val))
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        np.random.seed(seed + fold_idx)
        torch.manual_seed(seed + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + fold_idx)
        
        metrics = run_single_fold(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=config,
            device=device,
            fold_id=fold_idx,
            sparsity=sparsity,
            thresholding=thresholding,
            num_classes=num_classes,
            verbose=True
        )
        all_results.append(metrics)
    
    avg_results = {}
    for key in all_results[0].keys():
        if key == "per_class_f1":
            # Per-class F1 skorlarını ayrı işle
            all_per_class = {}
            for result in all_results:
                for class_name, f1_score in result["per_class_f1"].items():
                    if class_name not in all_per_class:
                        all_per_class[class_name] = []
                    all_per_class[class_name].append(f1_score)
            
            # Her sınıf için ortalama ve std hesapla
            avg_results["per_class_f1"] = {}
            avg_results["per_class_f1_std"] = {}
            for class_name, scores in all_per_class.items():
                avg_results["per_class_f1"][class_name] = np.mean(scores)
                avg_results["per_class_f1_std"][class_name] = np.std(scores)
        elif key != "loss":
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            avg_results[f"{key}_std"] = np.std(values)
    
    # Print summary
    print(f"\n  📊 {model_name} Ortalama Sonuçlar ({n_splits}-Fold CV):")
    print(f"     Accuracy:  {avg_results['accuracy']:.2f}% ± {avg_results['accuracy_std']:.2f}%")
    print(f"     Precision: {avg_results['precision']:.2f}% ± {avg_results['precision_std']:.2f}%")
    print(f"     Recall:    {avg_results['recall']:.2f}% ± {avg_results['recall_std']:.2f}%")
    print(f"     F1:        {avg_results['f1']:.2f}% ± {avg_results['f1_std']:.2f}%")
    
    # Print per-class F1 if available
    if "per_class_f1" in avg_results:
        print(f"\n  📈 Per-class F1 scores:")
        for class_name in ["W", "N1", "N2", "N3", "REM"]:
            if class_name in avg_results["per_class_f1"]:
                mean_f1 = avg_results["per_class_f1"][class_name]
                std_f1 = avg_results["per_class_f1_std"][class_name]
                print(f"     {class_name:>5}: {mean_f1:.2f}% ± {std_f1:.2f}%")
    
    return avg_results


def main():
    parser = argparse.ArgumentParser(description="Sleep Stage Classification K-Fold CV")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--output", type=str, default="results_kfold.json")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--channel", type=str, default="EEG Fpz-Cz",
                        choices=["EEG Fpz-Cz", "EEG Pz-Oz"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.max_subjects:
        config["data"]["max_subjects"] = args.max_subjects
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    # Sleep-EDF veri setini yükle
    print(f"\n{'#'*70}")
    print(f"# Sleep-EDF Veri Seti Yükleniyor")
    print(f"# Kanal: {args.channel}")
    print(f"# Yöntem: {args.n_splits}-Fold Stratified Cross Validation")
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
    
    # Sınıf dağılımı
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Sınıf dağılımı: {dict(zip(unique, counts))}")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_splits": args.n_splits,
        "method": "stratified_k_fold_cv",
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
        
        results = run_kfold_experiment(
            model_type=model_type,
            signals=signals,
            labels=labels,
            config=config,
            device=device,
            n_splits=args.n_splits,
            sparsity=sparsity if sparsity else 50,
            thresholding=thresholding if thresholding else "value",
            model_name=model_name,
            seed=args.seed
        )
        
        all_results["experiments"][model_name] = results
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Tüm deneyler tamamlandı! Sonuçlar '{args.output}' dosyasına kaydedildi.")
    print(f"{'='*70}")
    
    # Karşılaştırma tablosu
    print("\n" + "="*80)
    print("SONUÇ KARŞILAŞTIRMA TABLOSU")
    print("="*80)
    print("-"*70)
    print(f"{'Model':<30} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}")
    print("-"*70)
    for model_name, results in all_results["experiments"].items():
        acc = f"{results['accuracy']:.2f}±{results['accuracy_std']:.2f}"
        prec = f"{results['precision']:.2f}±{results['precision_std']:.2f}"
        rec = f"{results['recall']:.2f}±{results['recall_std']:.2f}"
        f1 = f"{results['f1']:.2f}±{results['f1_std']:.2f}"
        print(f"{model_name:<30} {acc:<15} {prec:<15} {rec:<15} {f1:<15}")


if __name__ == "__main__":
    main()
