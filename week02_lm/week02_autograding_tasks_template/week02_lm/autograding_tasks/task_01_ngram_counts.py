"""Task 01: n-gram counting."""

from collections import defaultdict, Counter

EOS = "_EOS_"
UNK = "_UNK_"


def count_ngrams(lines, n):
    """Count next-token frequencies for an n-gram language model.

    Args:
        lines: iterable of strings. Each string is a whitespace-tokenized line.
        n: n-gram order. Must be positive.

    Returns:
        A mapping from prefix tuples of length ``n - 1`` to mappings/counters
        of next-token counts. Each line should be treated as ending with EOS.
        Short prefixes should be left-padded with UNK.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be positive integer")

    counts = defaultdict(Counter)
    
    for line in lines:
        tokens = line.split()
        tokens.append(EOS)
        for i in range(len(tokens)):
            if i == 0:
                counts[tuple([UNK for _ in range(n - 1)])][tokens[i]] += 1
            elif i < n - 1:
                counts[tuple([UNK for _ in range(n - 1 - i)] + tokens[:i])][tokens[i]] += 1
            else:
                counts[tuple(tokens[i - n + 1:i])][tokens[i]] += 1
    
    return {prefix: dict(counter) for prefix, counter in counts.items()}
