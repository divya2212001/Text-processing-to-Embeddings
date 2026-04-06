import streamlit as st
from utils.cleaning import clean_text
from utils.tokenization import tokenize_text
from utils.normalization import normalize_text
from utils.vectorization import vectorize_text
from utils.embeddings import compute_similarity, get_embeddings, plot_clusters, plot_embeddings_pca, plot_similarity

st.set_page_config(page_title="Text → Embeddings Visualizer", layout="wide")

st.title("Text Preprocessing to Embeddings Visualizer")

# Sidebar
st.sidebar.header("Controls")

input_text = st.sidebar.text_area("Enter Text", "Natural Language Processing is amazing!")

run = st.sidebar.button("Run Processing")


lowercase = st.sidebar.checkbox("Lowercase", True)
remove_punct = st.sidebar.checkbox("Remove Punctuation", True)
remove_stopwords = st.sidebar.checkbox("Remove Stopwords", True)

token_type = st.sidebar.selectbox("Tokenization Type", ["word", "char"])
vector_type = st.sidebar.selectbox("Vectorization", ["BoW", "TF-IDF"])

if run:
    cleaned_text = clean_text(input_text, lowercase, remove_punct)
    tokens = tokenize_text(cleaned_text, token_type)
    normalized_tokens = normalize_text(tokens, remove_stopwords)

    # st.write("Processed Output:", normalized_tokens)

# Step 1: Cleaning
st.header("Text Cleaning & Normalization")

cleaned_text = clean_text(input_text, lowercase, remove_punct)

col1, col2 = st.columns(2)
col1.subheader("Original")
col1.write(input_text)

col2.subheader("Cleaned")
col2.write(cleaned_text)

st.info("Cleaning removes noise and standardizes text.")

# Step 2: Tokenization
st.header("Tokenization")

tokens = tokenize_text(cleaned_text, token_type)
st.write(tokens)

# Step 3: Stopwords + Lemmatization
st.header("Normalization")

normalized_tokens = normalize_text(tokens, remove_stopwords)
st.write(normalized_tokens)

# Step 4: Vocabulary
st.header("Vocabulary")

vocab = {word: idx for idx, word in enumerate(set(normalized_tokens))}
st.write(vocab)
st.write(f"Vocabulary Size: {len(vocab)}")

# Step 5: Vectorization
st.header("5️⃣ Vectorization")

if not normalized_tokens or " ".join(normalized_tokens).strip() == "":
    st.error("Empty vocabulary! All words were removed during preprocessing.")
    st.info("Try:")
    st.write("- Disable 'Remove Stopwords'")
    st.write("- Enter meaningful words like: 'AI is powerful'")
else:
    try:
        vectors, feature_names = vectorize_text(
            [" ".join(normalized_tokens)], vector_type
        )

        st.write("Feature Names:", feature_names)
        st.write("Vector:", vectors.toarray())

    except ValueError:
        st.error("Vectorization failed due to empty vocabulary.")

# Step 6: Word Embeddings
st.header("Word Embeddings")

embeddings = get_embeddings(normalized_tokens)

if embeddings:
    st.subheader("PCA Visualization")
    st.pyplot(plot_embeddings_pca(embeddings))

    st.subheader("Similarity")
    words, sim_matrix = compute_similarity(embeddings)
    st.pyplot(plot_similarity(words, sim_matrix))

    st.subheader("Clustering")
    st.pyplot(plot_clusters(embeddings))

# Step 7: Contextual Embeddings (Simplified)
st.header("Contextual Embeddings")

st.write("Same word in different contexts:")

sentence1 = "I went to the bank to deposit money"
sentence2 = "The river bank was beautiful"

st.write(sentence1)
st.write(sentence2)

st.info("Same word 'bank' has different meanings depending on context.")