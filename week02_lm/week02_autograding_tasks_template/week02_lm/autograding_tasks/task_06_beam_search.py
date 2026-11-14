"""Task 06: beam search generation."""

import numpy as np

BOS, EOS = ' ', '\n'

def generate_beamsearch(next_dist, prefix, beam_size=5, length=10):
    """Generate top hypotheses with beam search.

    Args:
        next_dist: callable that accepts a prefix string and returns
            ``{token: probability}``.
        prefix: initial text prefix.
        beam_size: number of hypotheses to keep.
        length: number of tokens to generate.

    Returns:
        A list of ``(text, log_probability)`` pairs sorted by score descending.
    """
    if not isinstance(beam_size, int) or beam_size <= 0:
        raise ValueError("beam_size must be positive integer")
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be positive integer")

    beam = [(prefix, 0.0)]
    epsilon = 1e-12

    for i in range(length):
        all_candidates = []
        for hyp in beam:
            candidates = next_dist(hyp[0])
            for token, prob in candidates.items():
                new_score = hyp[1] + np.log(prob)
                all_candidates.append((hyp[0] + token, new_score))
        all_candidates.sort(key=lambda p: p[1], reverse=True)
        beam = all_candidates[:beam_size]
    
    return sorted(beam, key=lambda p: p[1], reverse=True)
