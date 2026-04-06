import re
import string

def clean_text(text, lowercase=True, remove_punct=True):
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text