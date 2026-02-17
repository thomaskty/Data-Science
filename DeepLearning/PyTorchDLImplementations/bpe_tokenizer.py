#!/usr/bin/env python3
"""Minimal Byte Pair Encoding (BPE) tokenizer from scratch.

What this script does:
- Trains BPE merge rules on a tiny corpus
- Builds a subword vocabulary
- Encodes and decodes text

This is a teaching implementation (not optimized for speed).
"""

from collections import Counter, defaultdict


def get_stats(vocab):
    """Count frequency of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, v_in):
    """Merge a single pair in the vocabulary."""
    v_out = {}
    a, b = pair
    merged = a + b
    for word, freq in v_in.items():
        symbols = word.split()
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        v_out[" ".join(new_symbols)] = freq
    return v_out


class BPETokenizer:
    def __init__(self, merges, vocab, end_of_word="</w>"):
        self.merges = merges  # list of pairs
        self.vocab = vocab
        self.end_of_word = end_of_word

    @classmethod
    def train(cls, corpus, num_merges=50, end_of_word="</w>"):
        # Build word frequency vocab
        word_freq = Counter()
        for line in corpus:
            for word in line.strip().split():
                word_freq[word] += 1

        # Convert to symbol vocab (characters + </w>)
        vocab = {}
        for word, freq in word_freq.items():
            chars = list(word) + [end_of_word]
            vocab[" ".join(chars)] = freq

        merges = []
        for _ in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
            merges.append(best)
        return cls(merges, vocab, end_of_word=end_of_word)

    def _apply_merges(self, word):
        symbols = list(word) + [self.end_of_word]
        for a, b in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text):
        tokens = []
        for word in text.strip().split():
            symbols = self._apply_merges(word)
            # remove end-of-word marker for final tokens
            if symbols and symbols[-1] == self.end_of_word:
                symbols = symbols[:-1]
            tokens.extend(symbols)
        return tokens

    def decode(self, tokens):
        # Simple decode: join tokens and re-insert spaces by word boundary heuristic
        # Here we just join and add spaces where a token ends with </w>
        text = "".join(tokens)
        return text


def main():
    # Tiny synthetic corpus
    corpus = [
        "low lower lowest",
        "newer wider",
        "low lower",
        "newest lowest",
    ]

    tokenizer = BPETokenizer.train(corpus, num_merges=30)

    sample = "lower newest"
    tokens = tokenizer.encode(sample)

    print("Sample:", sample)
    print("Tokens:", tokens)
    print("Decoded:", tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
