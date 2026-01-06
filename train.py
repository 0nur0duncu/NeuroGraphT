import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from data import load_sleep_edf_dataset, SleepEDFDataset
from models import NeuroGraphT
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
from datetime import datetime


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config('config/config.yaml')
    
    data_config = config['data']
    train_config = config['training']
    class_weights_config = config['class_weights']
    
    signals, labels, _ = load_sleep_edf_dataset(
        data_dir=data_config['data_dir'],
        max_subjects=data_config['max_subjects'],
        verbose=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, 
        test_size=data_config['test_ratio'], 
        random_state=data_config['random_seed'], 
        stratify=labels
    )

    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    class_counts = np.bincount(y_train)
    
    print("\nSınıf Dağılımı (Training Set):")
    for i, (stage, count) in enumerate(zip(stage_names, class_counts)):
        print(f"  {stage}: {count:>5} ({count/len(y_train)*100:>5.1f}%)")

    class_weights = torch.FloatTensor(class_weights_config['weights']) if class_weights_config['enabled'] else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroGraphT(num_classes=data_config['num_classes']).to(device)

    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    print(f"\nModel eğitiliyor ({device})...")
    train_loader = DataLoader(SleepEDFDataset(X_train, y_train), batch_size=train_config['batch_size'], shuffle=True)
    model.train()

    num_epochs = train_config['num_epochs']
    epoch_losses = []  # Training loss'ları kaydet
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("\nTest ediliyor...")
    model.eval()
    test_loader = DataLoader(SleepEDFDataset(X_test, y_test), batch_size=64, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stage_names,
                yticklabels=stage_names, ax=ax1, cbar_kws={'label': 'Örnek Sayısı'})
    ax1.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
    ax1.set_ylabel('Gerçek Sınıf', fontsize=12)
    ax1.set_title('Confusion Matrix (Örnek Sayısı)', fontsize=14, fontweight='bold')

    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=stage_names,
                yticklabels=stage_names, ax=ax2, vmin=0, vmax=100, cbar_kws={'label': 'Yüzde (%)'})
    ax2.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
    ax2.set_ylabel('Gerçek Sınıf', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalize - %)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Training Loss Grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss - Epoch Bazında', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\n{'='*80}")
    print("SINIF BAZINDA DETAYLI METRİKLER")
    print(f"{'='*80}")

    for i, stage in enumerate(stage_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        total_samples = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{'='*80}")
        print(f"SINIF: {stage} (Toplam {total_samples} örnek)")
        print(f"{'='*80}")
        print(f"  Doğru Tahmin (TP): {tp:>6} örnek ({tp/total_samples*100:>5.1f}%)")
        print(f"  Yanlış Tahmin (FN): {fn:>6} örnek ({fn/total_samples*100:>5.1f}%)")
        print(f"\n  Precision: {precision*100:>5.1f}%")
        print(f"  Recall:    {recall*100:>5.1f}%")
        print(f"  F1 Score:  {f1*100:>5.1f}%")

        if fn > 0:
            print(f"\n  Yanlış Sınıflandırmalar:")
            misclassified = []
            for j, pred_stage in enumerate(stage_names):
                if i != j and cm[i, j] > 0:
                    misclassified.append((pred_stage, cm[i, j], cm[i, j]/total_samples*100))

            misclassified.sort(key=lambda x: x[1], reverse=True)
            for pred_stage, count, pct in misclassified:
                print(f"    → {count:>5} örnek {pred_stage} olarak sınıflandırıldı ({pct:>5.1f}%)")

        if fp > 0:
            print(f"\n  Yanlış Pozitifler:")
            false_positives = []
            for j, true_stage in enumerate(stage_names):
                if i != j and cm[j, i] > 0:
                    false_positives.append((true_stage, cm[j, i]))

            false_positives.sort(key=lambda x: x[1], reverse=True)
            for true_stage, count in false_positives:
                print(f"    ← {count:>5} {true_stage} örneği yanlışlıkla {stage} olarak tahmin edildi")

    print(f"\n{'='*80}")
    print("GENEL METRİKLER")
    print(f"{'='*80}")

    accuracy = np.trace(cm) / cm.sum() * 100
    total_samples = cm.sum()
    correct_predictions = np.trace(cm)
    wrong_predictions = total_samples - correct_predictions

    print(f"  Toplam Örnek:        {total_samples:>6}")
    print(f"  Doğru Tahmin:        {correct_predictions:>6} ({accuracy:.2f}%)")
    print(f"  Yanlış Tahmin:       {wrong_predictions:>6} ({100-accuracy:.2f}%)")

    f1_scores = []
    for i, stage in enumerate(stage_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) * 100
    print(f"  Macro F1 Score:      {macro_f1:>6.2f}%")
    print(f"{'='*80}")

    precisions = []
    recalls = []
    for i in range(len(stage_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision * 100)
        recalls.append(recall * 100)
    
    f1_scores_pct = [f * 100 for f in f1_scores]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(stage_names))
    width = 0.25
    
    ax1.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    ax1.bar(x, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width, f1_scores_pct, width, label='F1 Score', color='#2ecc71', alpha=0.8)
    ax1.set_xlabel('Sleep Stages', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('Sınıf Bazında Performance Metrikleri', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])
    
    colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    ax2.pie(class_counts, labels=stage_names, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax2.set_title('Training Set - Sınıf Dağılımı', fontsize=13, fontweight='bold')
    
    correct_per_class = [cm[i, i] for i in range(len(stage_names))]
    wrong_per_class = [cm[i, :].sum() - cm[i, i] for i in range(len(stage_names))]
    
    ax3.bar(x, correct_per_class, width*1.5, label='Doğru Tahmin', color='#2ecc71', alpha=0.8)
    ax3.bar(x, wrong_per_class, width*1.5, bottom=correct_per_class, label='Yanlış Tahmin', color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Sleep Stages', fontsize=12)
    ax3.set_ylabel('Örnek Sayısı', fontsize=12)
    ax3.set_title('Sınıf Bazında Doğru/Yanlış Tahmin Dağılımı', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stage_names)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    colors_f1 = ['#2ecc71' if f > 70 else '#f39c12' if f > 50 else '#e74c3c' for f in f1_scores_pct]
    bars = ax4.barh(stage_names, f1_scores_pct, color=colors_f1, alpha=0.8)
    ax4.set_xlabel('F1 Score (%)', fontsize=12)
    ax4.set_ylabel('Sleep Stages', fontsize=12)
    ax4.set_title('Sınıf Bazında F1 Score Karşılaştırması', fontsize=13, fontweight='bold')
    ax4.set_xlim([0, 105])
    ax4.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores_pct)):
        ax4.text(score + 1, i, f'{score:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"neurographt_full_dataset_{timestamp}.pth"
    model_path = os.path.join(checkpoint_dir, model_filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_weights': class_weights.numpy() if class_weights is not None else None,
        'num_epochs': num_epochs,
        'learning_rate': train_config['learning_rate'],
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'per_class_f1': {stage: f1*100 for stage, f1 in zip(stage_names, f1_scores)},
        'training_samples': len(y_train),
        'test_samples': len(y_test),
        'timestamp': timestamp
    }

    torch.save(checkpoint, model_path)
    print(f"\n✓ Model kaydedildi: {model_path}")


if __name__ == "__main__":
    main()
