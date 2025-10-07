from sentence_transformers import SentenceTransformer, util
import nltk
import os

# Ensure NLTK tokenizer data is available
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Fix NLTK data path for environments with custom paths
nltk.data.path.append(os.path.expanduser(r"~\nltk_data"))

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = []
corpus_embeddings = None


def build_corpus(paragraph):
    """Split paragraph into sentences."""
    sentences = sent_tokenize(paragraph)
    return sentences


def update_corpus(paragraph):
    """Add new sentences to corpus and update embeddings."""
    global corpus, corpus_embeddings
    new_sentences = build_corpus(paragraph)
    corpus.extend(new_sentences)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    print(f"✅ Corpus updated. Total sentences: {len(corpus)}")


def search(query, top_k=3):
    """Search corpus for top_k most semantically similar sentences."""
    if not corpus_embeddings:
        print("❌ Corpus is empty. Please add text first.")
        return

    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = cosine_scores.argsort(descending=True)[:top_k]

    print(f"\n🔍 Query: {query}\n")
    for idx in top_results:
        print(f"{corpus[idx]}  (Score: {cosine_scores[idx]:.4f})")


if __name__ == "__main__":
    print("📌 Semantic Search Tool")
    print("Type 'add' to add corpus text, 'search' to search, 'exit' to quit.\n")

    while True:
        command = input("Enter command (add/search/exit): ").strip().lower()

        if command == "exit":
            print("👋 Exiting Semantic Search Tool.")
            break
        elif command == "add":
            paragraph = input("\nPaste your paragraph/text:\n")
            update_corpus(paragraph)
        elif command == "search":
            query = input("\nEnter your search query:\n")
            search(query)
        else:
            print("❌ Unknown command. Please use 'add', 'search', or 'exit'.")
