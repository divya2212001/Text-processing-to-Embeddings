TEXT PREPROCESSING TO EMBEDDINGS
 “How does raw text get transformed into meaningful numerical representations that models can understand? Build an interactive system to visualize how text is cleaned, tokenized, and converted into embeddings.”
Students must design a Streamlit app that allows users to explore how raw text is processed step-by-step into vector representations used in machine learning models.

KEY CONCEPTUAL COMPONENTS (7 CORE ELEMENTS)
1. Text Cleaning & Normalization
 Concept
 Raw text is cleaned to remove noise and standardize format:
 Lowercasing
 Removing punctuation
 Removing special characters
Role in Model
 Improves consistency and reduces irrelevant variation in text.
Visualization
 Show original vs cleaned text
 Highlight removed elements
 Toggle different preprocessing steps
AI-Assisted Implementation
 Apply cleaning functions dynamically
 Allow user-controlled preprocessing options

2. Tokenization
 Concept
 Text is split into smaller units called tokens:
 Words
 Subwords
 Characters
Role in Model
 Breaks text into manageable pieces for further processing.
Visualization
 Display tokens from input text
 Compare word-level vs subword tokenization
 Highlight token boundaries
AI-Assisted Implementation
 Implement different tokenization strategies
 Allow switching between them

3. Stopword Removal & Stemming/Lemmatization
 Concept
 Common words may be removed, and words can be reduced to base forms:
 Stopword removal
 Stemming
 Lemmatization
Role in Model
 Reduces dimensionality and focuses on meaningful content.
Visualization Ideas
 Show removed stopwords
 Compare original vs stemmed/lemmatized text
 Highlight transformations
AI-Assisted Implementation
 Apply NLP preprocessing techniques
 Display transformations interactively

4. Vocabulary Building
 Concept
 A vocabulary is created mapping tokens to indices.
Role in Model
 Defines the space of possible inputs for numerical encoding.
Visualization Ideas
 Display token-to-index mapping
 Show vocabulary size
 Highlight most frequent tokens
AI-Assisted Implementation
 Build vocabulary dynamically
 Sort tokens by frequency

5. Text Vectorization (Bag-of-Words / TF-IDF)
 Concept
 Convert tokens into numerical vectors:
 Bag-of-Words (counts)
 TF-IDF (importance-weighted counts)
Role in Model
 Transforms text into a format usable by traditional ML models.
Visualization
 Show vector representation of a sentence
 Compare BoW vs TF-IDF
 Display sparse vector structure
AI-Assisted Implementation
 Compute BoW and TF-IDF vectors
 Visualize values in table or heatmap

6. Word Embeddings (Dense Representations)
 Concept
 Words are mapped to dense vectors capturing semantic meaning:
 Word2Vec
 GloVe
 Learned embeddings
Role in Model
 Capture relationships between words in a continuous space.
Visualization
 Plot embeddings in 2D (e.g., PCA)
 Show similarity between words
 Highlight clusters of related words
AI-Assisted Implementation
 Load or train embeddings
 Compute similarity scores
 Visualize embedding space

7. Contextual Embeddings
 Concept
 Modern models generate context-dependent embeddings:
 Same word → different meaning in different sentences
Role in Model
 Captures nuanced meaning based on context.
Visualization
 Compare embeddings of same word in different sentences
 Highlight context influence
 Show attention-based differences
AI-Assisted Implementation
 Use transformer-based embeddings
 Visualize contextual differences

STREAMLIT APP STRUCTURE
1. Sidebar Controls
 Input text field
 Toggle:
 Lowercase
 Remove punctuation
 Remove stopwords
Dropdown:
 Tokenization type
 Vectorization method

2. Main Sections (Tabs or Expanders)
Tab 1: Text Cleaning
 Show preprocessing steps
Tab 2: Tokenization
 Display tokens
Tab 3: Text Normalization
 Show stemming/lemmatization
Tab 4: Vocabulary
 Display token-index mapping
Tab 5: Vectorization
 Show BoW / TF-IDF vectors
Tab 6: Word Embeddings
 Visualize dense vectors
Tab 7: Contextual Embeddings
 Compare context-based representations

3. Explanation Layer
 Each section should include:
 2–3 line intuitive explanation
 Key takeaway: “What should you observe?”

PROJECT WORKFLOW
Input raw text
 Apply cleaning and normalization
 Tokenize text
 Remove stopwords and normalize words
 Build vocabulary
 Convert text to numerical vectors (BoW/TF-IDF)
 Generate word embeddings
 Explore contextual embeddings
 Visualize transformations at each step
 Polish UI with Streamlit layout
