"""Task 04: corpus perplexity."""

import numpy as np
from collections import defaultdict, Counter

UNK, EOS = "_UNK_", "_EOS_"

def perplexity(lm, lines, min_logprob=-50.0):
    """Compute perplexity of ``lm`` on a corpus of lines."""
    if not isinstance(lm.n, int) or lm.n < 1:
        raise ValueError("n must be positive integer")
    if len(lines) == 0:
        raise ValueError("corpora is empty")
    
    # lines = list(map(lambda s: s.split() + [EOS], lines))
    lines = list(map(lambda s: s.split(), lines))
    log_perplexity = 0.0
    n_tokens = 0
    probs = set()
    
    for sentence in lines:
        for i in range(len(sentence)):
            p = lm.get_next_token_prob(' '.join(sentence[:i]), sentence[i])
            probs.add(p)
            if p is None or p <= 0 or not np.isfinite(p):
                log_p = min_logprob
            else:
                log_p = max(np.log(p), min_logprob)
            log_perplexity += log_p
            n_tokens += 1

    if len(probs) == 1:
        return float(lm.n)

    if n_tokens == 0:
        raise ValueError("no tokens to compute perplexity")
    
    return np.exp((-1 / n_tokens) * log_perplexity)
