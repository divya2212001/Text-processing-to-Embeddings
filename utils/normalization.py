import spacy

nlp = spacy.load("en_core_web_sm")

def normalize_text(tokens, remove_stopwords=True):
    doc = nlp(" ".join(tokens))
    result = []
    
    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        result.append(token.lemma_)
    
    return result