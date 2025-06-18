# 🧠 Text Summarization App (Graph-based Extractive)

A Streamlit-based web application for **extractive text summarization** using **TextRank algorithm** with **GloVe word embeddings** and **cosine similarity**.

📁 **GitHub Repo**: [GitHub Repo Link](https://github.com/gourab-9/Text-summarization)
---

## 📌 Features

- Accepts raw text input from the user.
- Uses **NLTK** for sentence tokenization and stopword removal.
- Computes sentence embeddings using **GloVe (100d)** vectors.
- Builds a sentence similarity graph and applies **PageRank (TextRank)**.
- Allows users to select the number of sentences in the summary.
- Lightweight and fast — perfect for extractive summarization tasks.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI framework
- **NLTK** – Natural language processing (tokenization, stopwords)
- **GloVe** – Pre-trained word embeddings
- **Scikit-learn** – Cosine similarity
- **NetworkX** – PageRank graph algorithm
- **Pandas / NumPy** – Data handling

---

## 🧪 How It Works

1. Tokenize input text into sentences.
2. Preprocess each sentence (remove non-alphabetic, lowercase, remove stopwords).
3. Convert each sentence into a vector using GloVe word embeddings.
4. Calculate pairwise cosine similarity matrix.
5. Construct a graph and rank sentences using **PageRank**.
6. Return the top N sentences as the summary.

---

## 🧳 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/gourab-9/text-summarizer-streamlit.git
   cd text-summarizer-streamlit
   ```
2. **Download GloVe Embeddings**
`Download glove.6B.100d.txt` from GloVe Website
and place it in the root directory.

3. **Create environment and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app**
  ```bash
streamlit run app.py
```
## 📂 Files
- `app.py` – Streamlit app with summarization logic
- `glove.6B.100d.txt` – Pre-trained GloVe word vectors (100d)
- `requirements.txt` – All dependencies listed
- (Optional) `tennis.csv` – Example dataset (not used in final UI)

## 📋 Example
Input:
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.

Output Summary:
Natural language processing is a subfield of linguistics, computer science, and AI that focuses on interactions between computers and human language.

