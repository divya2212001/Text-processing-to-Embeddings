import nltk
from nltk.tokenize import word_tokenize

def tokenize_text(text, token_type="word"):
    if token_type == "word":
        return word_tokenize(text)
    elif token_type == "char":
        return list(text)