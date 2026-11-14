"""Task 03: temperature and nucleus sampling."""

import numpy as np

def sample_with_temperature(dist, temperature=1.0, rng=None):
    """Sample one token from a probability distribution with temperature."""
    tokens = list(dist.keys())
    probs = list(dist.values())

    if len(tokens) == 0:
        raise ValueError("empty distribution")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0:
        return tokens[np.argmax(probs)]
    elif temperature == 1:
        return np.random.choice(tokens, p=probs)

    # prob_sum = sum([p ** (1 / temperature) for p in probs])
    # new_probs = [p ** (1 / temperature) / prob_sum for p in probs]
    logits = np.log(np.maximum(probs, 1e-300)) / temperature
    logits -= logits.max()
    new_probs = np.exp(logits)
    new_probs /= new_probs.sum()

    if rng is None:
        rng = np.random

    return rng.choice(tokens, p=new_probs)


def nucleus_filter(dist, nucleus=0.9):
    """Keep the smallest high-probability token set with cumulative mass >= nucleus."""
    tokens = list(dist.keys())
    probs = list(dist.values())

    if len(tokens) == 0:
        raise ValueError("empty distribution")
    if nucleus <= 0 or nucleus > 1:
        raise ValueError("nucleus value must be positive and <= 1")

    dist_list = list(zip(tokens, probs))
    dist_list.sort(key=lambda p: p[1], reverse=True)
    cumulative_sum = 0.0
    tokens = []

    if nucleus == 1.0:
        return dist_list

    for token, prob in dist_list:
        tokens.append((token, prob))
        cumulative_sum += prob
        if cumulative_sum >= nucleus:
            break

    if not tokens:
        tokens.append(dist_list[0])
    
    return tokens

def sample_nucleus(dist, nucleus=0.9, rng=None):
    """Sample one token after applying nucleus filtering."""
    candidates = nucleus_filter(dist, nucleus)
    tokens = list(map(lambda p: p[0], candidates))
    probs = list(map(lambda p: p[1], candidates))
    prob_sum = sum(probs)
    probs = list(map(lambda p: p / prob_sum, probs))

    if len(tokens) == 0:
        raise ValueError("empty distribution")
    if nucleus <= 0 or nucleus > 1:
        raise ValueError("nucleus value must be positive and <= 1")

    if rng is None:
        rng = np.random
    return rng.choice(tokens, p=probs)
