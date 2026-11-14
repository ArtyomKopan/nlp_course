"""Task 02: n-gram language model."""

from collections import defaultdict, Counter

UNK, EOS = "_UNK_", "_EOS_"

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

class NGramLanguageModel:
    def __init__(self, lines, n):
        """
        Train a simple count-based language model:
        compute probabilities P(w_t | prefix) given ngram counts

        :param n: computes probability of next token given (n - 1) previous words
        :param lines: an iterable of strings with space-separated tokens
        """

        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be positive integer")
        
        self.n = n

        counts = count_ngrams(lines, self.n)

        # compute token proabilities given counts
        self.probs = defaultdict(Counter)
        # probs[(word1, word2)][word3] = P(word3 | word1, word2)

        # populate self.probs with actual probabilities
        for prefix, words_list in counts.items():
            word_count_sum = sum(words_list.values())
            for word in words_list.keys():
                self.probs[prefix][word] = words_list[word] / word_count_sum

    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {token : it's probability} for all tokens with positive probabilities
        """
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]
        prefix = [ UNK ] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]

    def get_next_token_prob(self, prefix, next_token):
        """
        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :returns: P(next_token|prefix) a single number, 0 <= P <= 1
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)
