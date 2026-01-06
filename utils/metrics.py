import numpy as np
import torch
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def calculate_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return accuracy_score(y_true, y_pred) * 100


def calculate_precision(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return precision_score(
        y_true, y_pred, average=average, zero_division=0
    ) * 100

def calculate_recall(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return recall_score(
        y_true, y_pred, average=average, zero_division=0
    ) * 100


def calculate_f1(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return f1_score(
        y_true, y_pred, average=average, zero_division=0
    ) * 100


def calculate_per_class_f1(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    # Calculate per-class F1 scores
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0) * 100
    
    # Create class names if not provided
    if class_names is None:
        n_classes = len(np.unique(y_true))
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Create dictionary mapping class names to F1 scores
    per_class_f1 = {name: score for name, score in zip(class_names, f1_scores)}
    
    return per_class_f1


def calculate_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro",
    class_names: Optional[List[str]] = None,
    include_per_class: bool = False
) -> Dict[str, Union[float, Dict[str, float]]]:

    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred, average),
        "recall": calculate_recall(y_true, y_pred, average),
        "f1": calculate_f1(y_true, y_pred, average)
    }
    
    if include_per_class:
        metrics["per_class_f1"] = calculate_per_class_f1(y_true, y_pred, class_names)
    
    return metrics

def get_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return confusion_matrix(y_true, y_pred)


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


class MetricTracker:
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["loss", "accuracy", "precision", "recall", "f1"]
        self.history: Dict[str, List[float]] = {m: [] for m in self.metrics}
        self.best: Dict[str, float] = {}
    
    def update(self, values: Dict[str, float]):
        for name, value in values.items():
            if name in self.history:
                self.history[name].append(value)
                if name == "loss":
                    if name not in self.best or value < self.best[name]:
                        self.best[name] = value
                else:
                    if name not in self.best or value > self.best[name]:
                        self.best[name] = value
    
    def get_last(self, name: str) -> Optional[float]:
        if name in self.history and self.history[name]:
            return self.history[name][-1]
        return None
    
    def get_best(self, name: str) -> Optional[float]:
        return self.best.get(name)
    
    def get_history(self, name: str) -> List[float]:
        return self.history.get(name, [])
    
    def reset(self):
        self.history = {m: [] for m in self.metrics}
        self.best = {}
    
    def summary(self) -> str:
        lines = []
        for name in self.metrics:
            last = self.get_last(name)
            best = self.get_best(name)
            if last is not None:
                lines.append(f"{name}: {last:.2f} (best: {best:.2f})")
        return " | ".join(lines)
