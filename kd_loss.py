import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """
    Knowledge-distillation loss combining a soft-target distillation term with a
    hard-label cross-entropy term (Hinton et al., 2015).

    Args:
        student_logits: (N, C) raw (un-normalised) student logits.
        teacher_logits: (N, C) raw (un-normalised) teacher logits/similarity scores.
        labels: (N,) ground-truth class indices (LongTensor).
        temperature: float > 0, softmax temperature for the distillation term.
        alpha: float in [0, 1], weight on the distillation term
               (the hard-label term is weighted (1 - alpha)).

    Returns:
        A scalar torch.Tensor: the combined loss.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    distillation_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature ** 2)
    hard_label_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation_loss + (1 - alpha) * hard_label_loss
