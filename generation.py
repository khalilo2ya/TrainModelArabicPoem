import tensorflow as tf
import numpy as np

def generate_text(model, start_string, vectorizer, max_length=1000):
    """Generate text using a language generation model.

    Arguments:
    model -- a Keras model for language generation
    start_string -- a string to use as the starting point for text generation
    vectorizer -- a vectorizer object used to vectorize the text
    max_length -- an integer representing the maximum length of the generated text (defaults to 1000)

    Returns:
    generated_text -- a string representing the generated text
    """
    num_generate = max_length
    input_eval = [vectorizer.word_index[s] for s in start_string.split()]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        word = vectorizer.index_word[predicted_id]
        text_generated.append(word)

    return ' '.join(text_generated)
