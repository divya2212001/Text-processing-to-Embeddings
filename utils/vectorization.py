from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize_text(texts, method="BoW"):
    if method == "BoW":
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer.get_feature_names_out()