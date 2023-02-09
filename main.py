import preprocessing
import tokenization
import vectorization
import model
import training
import generation

def main():
    # Load the poems from file
    poems = preprocessing.load_poems("data/poems.txt")
    
    # Preprocess the poems
    preprocessed_poems = preprocessing.preprocess(poems)
    
    vocab_size = 20
    # Tokenize the poems
    tokenized_poems = tokenization.tokenize(preprocessed_poems, vocab_size)
    # Vectorize the poems
    # vectorized_poems = vectorization.vectorize(tokenized_poems)
    vectorized_poems = vectorization.vectorize(tokenized_poems, vocab_size)

    
    # Define and compile the model
    model = model.define_model()
    
    # Train the model
    training.train(model, vectorized_poems)
    
    # Generate a poem using the trained model
    generated_poem = generation.generate_poem(model, vectorized_poems)
    
    # Print the generated poem
    print("Generated Poem:")
    print(generated_poem)

if __name__ == "__main__":
    main()
