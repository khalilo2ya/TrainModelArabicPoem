import tensorflow as tf

def build_model(vocab_size, seq_length, embedding_dim=64, rnn_units=64):
    """Build a language generation model.

    Arguments:
    vocab_size -- an integer representing the size of the vocabulary
    seq_length -- an integer representing the length of a sequence
    embedding_dim -- an integer representing the dimensionality of the embedding layer (defaults to 64)
    rnn_units -- an integer representing the number of units in the RNN layer (defaults to 64)

    Returns:
    model -- a Keras model for language generation
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[None, seq_length]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model
