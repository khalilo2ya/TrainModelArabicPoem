import numpy as np

def one_hot_encode(sequence, vocab_size):
    """One-hot encode a sequence of integers.

    Arguments:
    sequence -- a list of integers representing a tokenized poem
    vocab_size -- an integer representing the size of the vocabulary

    Returns:
    encoded_sequence -- a numpy array of shape (sequence length, vocab_size) representing the one-hot encoded sequence
    """
    encoded_sequence = np.zeros((len(sequence), vocab_size))
    for i, word_index in enumerate(sequence):
        print(f"word_index: {word_index}, type(word_index): {type(word_index)}")

        encoded_sequence[i, word_index] = 1

    return encoded_sequence

def vectorize(tokenized_poems, vocab_size):
    """Vectorize the tokenized poems.

    Arguments:
    tokenized_poems -- a list of lists of integers, where each list of integers is a tokenized poem
    vocab_size -- an integer representing the size of the vocabulary

    Returns:
    vectorized_poems -- a numpy array of shape (number of poems, sequence length, vocab_size) representing the vectorized poems
    """
    vectorized_poems = []
    for poem in tokenized_poems:
        encoded_poem = one_hot_encode(poem, vocab_size)
        vectorized_poems.append(encoded_poem)

    return np.array(vectorized_poems)
