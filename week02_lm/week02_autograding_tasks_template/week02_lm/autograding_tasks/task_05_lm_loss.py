"""Task 05: masked language-model loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_mask(input_ix, eos_ix):
    """Return a boolean mask that keeps tokens up to and including first EOS."""
    return F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True)


def compute_lm_loss(logits, targets, eos_ix):
    """Compute cross-entropy language-model loss, ignoring tokens after EOS."""
    mask = compute_mask(targets, eos_ix)
    
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction='none'
    ).reshape(targets.shape)
    
    loss = (loss_per_token * mask).sum() / mask.sum()
    
    return loss
