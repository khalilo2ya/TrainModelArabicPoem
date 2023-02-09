import tensorflow as tf
import numpy as np

def sample(preds, temperature=1.0):
    """Sample an index from a probability array.

    Arguments:
    preds -- a 1-D numpy array representing the predicted probabilities
    temperature -- a float controlling the randomness of the sampling (defaults to 1.0)

    Returns:
    index -- an integer representing the sampled index
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, start_string, vectorizer, max_length=1000, temperature=1.0):
    """Generate text using a language generation model.

    Arguments:
    model -- a Keras model for language generation
    start_string -- a string to use as the starting point for text generation
    vectorizer -- a vectorizer object used to vectorize the text
    max_length -- an integer representing the maximum length of the generated text (defaults to 1000)
    temperature -- a float controlling the randomness of the generated text (defaults to 1.0)

    Returns:
    generated_text -- a string representing the generated text
    """
    input_eval = [vectorizer.word_index[s] for s in start_string.split()]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for i in range(max_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = sample(predictions, temperature)
        input_eval = tf.expand_dims([predicted_id], 0)
        word = vectorizer.index_word[predicted_id]
        text_generated.append(word)

    return ' '.join(text_generated)
