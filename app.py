import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os

# Download required NLTK resources
nltk.download()
nltk.download('punkt')
nltk.download('stopwords')

# Load GloVe embeddings only once
@st.cache_resource
def load_glove_embeddings():
    embeddings = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings

# Load example dataset
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/tennis.csv')
    return df

# Text Summarization Function
def text_summarization(text, num_sentences=5):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "No valid sentences found for summarization."

    word_embeddings = load_glove_embeddings()

    # Clean sentences
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ", regex=True)
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = set(stopwords.words('english'))
    
    def remove_stopwords(sen):
        return " ".join([word for word in sen if word not in stop_words])
    
    clean_sentences = [remove_stopwords(s.split()) for s in clean_sentences]

    # Create sentence vectors
    sentence_vector = []
    for sentence in clean_sentences:
        if sentence:
            v = sum([word_embeddings.get(word, np.zeros((100,))) for word in sentence.split()]) / (len(sentence.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vector.append(v)

    # Similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(
                    sentence_vector[i].reshape(1, 100),
                    sentence_vector[j].reshape(1, 100)
                )[0, 0]

    # Ranking sentences
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary = ""
    for i in range(min(num_sentences, len(ranked_sentences))):
        summary += ranked_sentences[i][1] + " "

    return summary.strip()

# Streamlit Web App
def main():
    st.title("Text Summarization App (Graph-based Extractive)")

    # Load optional sample data
    # if st.checkbox("Show sample tennis data"):
    #     df = load_data()
    #     st.dataframe(df.head())

    # Text input
    input_text = st.text_area("Enter your text to summarize:", height=300)

    # Summary length slider
    num_sentences = st.slider("Number of sentences in summary:", 1, 10, 5)

    # Button to trigger summarization
    if st.button("Summarize Now"):
        if input_text.strip():
            summary = text_summarization(input_text, num_sentences)
            st.subheader("📝 Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
