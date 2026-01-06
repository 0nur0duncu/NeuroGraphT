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
from datetime import datetime


def main():
    data_path = "dataset/sleep-edfx"
    
    signals, labels, _ = load_sleep_edf_dataset(
        data_dir=data_path,
        max_subjects=None,
        verbose=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, test_size=0.3, random_state=42, stratify=labels
    )

    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    class_counts = np.bincount(y_train)
    
    print("\nSınıf Dağılımı (Training Set):")
    for i, (stage, count) in enumerate(zip(stage_names, class_counts)):
        print(f"  {stage}: {count:>5} ({count/len(y_train)*100:>5.1f}%)")

    class_weights = torch.FloatTensor([1.0, 15.0, 4.0, 20.0, 12.0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroGraphT(num_classes=5).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print(f"\nModel eğitiliyor ({device})...")
    train_loader = DataLoader(SleepEDFDataset(X_train, y_train), batch_size=64, shuffle=True)
    model.train()

    num_epochs = 20
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

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"neurographt_full_dataset_{timestamp}.pth"
    model_path = os.path.join(checkpoint_dir, model_filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_weights': class_weights.numpy(),
        'num_epochs': num_epochs,
        'learning_rate': 0.0005,
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
