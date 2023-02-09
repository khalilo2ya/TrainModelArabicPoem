import re

def load_poems(file_path):
    """Load the poems from the given file.

    Arguments:
    file_path -- the path to the file containing the poems

    Returns:
    poems -- a list of strings, where each string is a poem
    """
    with open(file_path, "r") as f:
        poems = f.readlines()
    return poems

def preprocess(poems):
    """Preprocess the poems by removing any unwanted characters and normalizing the text.

    Arguments:
    poems -- a list of strings, where each string is a poem

    Returns:
    preprocessed_poems -- a list of strings, where each string is a preprocessed poem
    """
    preprocessed_poems = []
    for poem in poems:
        # Remove any unwanted characters
        cleaned_poem = clean_poem(poem)
        
        # Normalize the text
        normalized_poem = normalize_text(cleaned_poem)
        
        preprocessed_poems.append(normalized_poem)
    return preprocessed_poems

def clean_poem(poem):
    """Clean the given poem by removing any unwanted characters.

    Arguments:
    poem -- a string representing the poem

    Returns:
    cleaned_poem -- a string representing the cleaned poem
    """
    # Remove any unwanted characters (e.g., punctuation, numbers, special characters)
    cleaned_poem = re.sub(r'[^\w\s]', '', poem)
    
    return cleaned_poem

def normalize_text(text):
    """Normalize the given text.

    Arguments:
    text -- a string representing the text to be normalized

    Returns:
    normalized_text -- a string representing the normalized text
    """
    # Lowercase the text
    normalized_text = text.lower()
    
    return normalized_text
