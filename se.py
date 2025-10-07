from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = []
corpus_embeddings = None


def build_corpus(paragraph):
    sentences = sent_tokenize(paragraph)
    return sentences


def update_corpus(paragraph):
    global corpus, corpus_embeddings
    new_sentences = build_corpus(paragraph)
    corpus.extend(new_sentences)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    print(f" Corpus updated. Total sentences: {len(corpus)}")


def search(query, top_k=3):
    if not corpus_embeddings:
        print(" Corpus is empty. Please add text first.")
        return

    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = cosine_scores.argsort(descending=True)[:top_k]

    print(f"\n Query: {query}\n")
    for idx in top_results:
        print(f"{corpus[idx]}  (Score: {cosine_scores[idx]:.4f})")


if __name__ == "__main__":
    print(" Semantic Search Tool")
    print("Type 'add' to add corpus text, 'search' to search, 'exit' to quit.\n")

    while True:
        command = input("Enter command (add/search/exit): ").strip().lower()

        if command == "exit":
            break
        elif command == "add":
            paragraph = input("\nPaste your paragraph/text:\n")
            update_corpus(paragraph)
        elif command == "search":
            query = input("\nEnter your search query:\n")
            search(query)
        else:
            print(" Unknown command. Please use 'add', 'search', or 'exit'.")
