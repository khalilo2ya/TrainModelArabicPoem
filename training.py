import tensorflow as tf

def train(model, vectorized_poems, epochs=100, batch_size=128, save_model=False, model_file='model.h5'):
    """Train a language generation model.

    Arguments:
    model -- a Keras model for language generation
    vectorized_poems -- a numpy array of shape (number of poems, sequence length, vocab_size) representing the vectorized poems
    epochs -- an integer representing the number of training epochs (defaults to 100)
    batch_size -- an integer representing the size of a training batch (defaults to 128)
    save_model -- a boolean indicating whether to save the trained model (defaults to False)
    model_file -- a string representing the file name to save the model (defaults to 'model.h5')

    Returns:
    history -- a History object containing the training history
    """
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    history = model.fit(vectorized_poems, vectorized_poems, epochs=epochs, batch_size=batch_size)

    if save_model:
        model.save(model_file)

    return history
