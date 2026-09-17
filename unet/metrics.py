"""
Dice Similarity Coefficient (DSC) computation for multi-class segmentation.

DSC for a single class = 2 * |prediction ∩ ground_truth| / (|prediction| + |ground_truth|)
                        = 2 * TP / (2*TP + FP + FN)

Computed per class so each of the 4 OASIS segmentation labels can be reported
and checked against the assignment's >0.9 DSC requirement individually, not
just as an average that could hide a poorly-performing class.
"""

import torch


def dice_score_per_class(pred_logits, target, num_classes, eps=1e-6):
    """
    pred_logits: [B, num_classes, H, W] raw model output (before softmax)
    target: [B, H, W] integer class labels

    Returns: list of length num_classes, DSC for each class averaged over the batch.
    """
    pred_classes = torch.argmax(pred_logits, dim=1)  # [B, H, W]

    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.mean().item())

    return dice_scores


def dice_loss(pred_logits, target, num_classes, eps=1e-6):
    """
    Differentiable soft Dice loss (1 - mean Dice), computed from softmax
    probabilities rather than argmax, so gradients can flow. Useful either as
    the sole training loss or combined with CrossEntropyLoss
    (loss = CE + dice_loss) - the combination often gives better boundary
    accuracy than CE alone, which is exactly what DSC measures.
    """
    pred_probs = torch.softmax(pred_logits, dim=1)  # [B, C, H, W]

    # One-hot encode the target for elementwise multiplication with pred_probs
    target_onehot = torch.nn.functional.one_hot(target, num_classes)  # [B, H, W, C]
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()          # [B, C, H, W]

    intersection = (pred_probs * target_onehot).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice_per_class = (2.0 * intersection + eps) / (union + eps)  # [B, C]
    return 1.0 - dice_per_class.mean()
