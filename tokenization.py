import numpy as np

def tokenize(poems, vocab_size):
    """Tokenize the poems into sequences of integers.

    Arguments:
    poems -- a list of strings, where each string is a preprocessed poem
    vocab_size -- an integer representing the size of the vocabulary to use for tokenization

    Returns:
    tokenized_poems -- a list of lists of integers, where each list of integers is a tokenized poem
    word_to_index -- a dictionary mapping words to their corresponding index in the vocabulary
    index_to_word -- a dictionary mapping indices to their corresponding word in the vocabulary
    """
    # Create a list of all the words in the poems
    all_words = []
    for poem in poems:
        words = poem.split()
        all_words.extend(words)

    # Count the frequency of each word
    word_counts = {}
    for word in all_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Sort the words by frequency
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Use the `vocab_size` most frequent words as the vocabulary
    vocabulary = [word for word, count in sorted_word_counts[:vocab_size]]

    # Create a dictionary mapping words to their corresponding index in the vocabulary
    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    # Create a dictionary mapping indices to their corresponding word in the vocabulary
    index_to_word = {index: word for index, word in enumerate(vocabulary)}

    # Tokenize the poems into sequences of integers
    tokenized_poems = []
    for poem in poems:
        words = poem.split()
        tokenized_poem = [word_to_index[word] for word in words if word in word_to_index]
        tokenized_poems.append(tokenized_poem)

    return tokenized_poems, word_to_index, index_to_word

def pad_sequences(tokenized_poems, max_length):
    """Pad the tokenized poems with zeros to ensure they all have the same length.

    Arguments:
    tokenized_poems -- a list of lists of integers, where each list of integers is a tokenized poem
    max_length -- an integer representing the maximum length of the tokenized poems

    Returns:
    padded_poems -- a numpy array of shape (number of poems, max_length) representing the padded poems
    """
    padded_poems = []
    for poem in tokenized_poems:
        if len(poem) < max_length:
            padding = [0 for _ in range(max_length - len(poem))]
            padded_poem = poem + padding
        else:
            padded_poem = poem[:max_length]
        padded_poems.append(padded_poem)

    return np.array(padded_poems)
