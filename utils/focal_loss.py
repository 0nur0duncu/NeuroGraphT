"""
Focal Loss implementation for handling severe class imbalance.
Focuses more on hard-to-classify samples (like N1 stage).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Class weights (list or tensor), shape [num_classes]
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities for each sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = focal_term * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples"
    https://arxiv.org/abs/1901.05555
    
    Args:
        samples_per_class: Number of samples for each class [num_classes]
        beta: Balancing parameter (0.99, 0.999, or 0.9999)
        gamma: Focal loss gamma parameter
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        
        self.focal_loss = FocalLoss(alpha=weights.tolist(), gamma=gamma)
    
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)
