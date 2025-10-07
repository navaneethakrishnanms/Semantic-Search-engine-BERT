from sentence_transformers import SentenceTransformer, util
import nltk
import os


print("Ensuring NLTK 'punkt' package is available...")
nltk.download('punkt', quiet=True)
print("NLTK check complete.")

from nltk.tokenize import sent_tokenize

print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")



corpus = []
corpus_embeddings = None


def update_corpus(paragraph: str):
    
    global corpus, corpus_embeddings
    
   
    new_sentences = sent_tokenize(paragraph)
    if not new_sentences:
        print(" No new sentences found in the provided text.")
        return
        
    
    corpus.extend(new_sentences)
    
    
    print("Encoding corpus... This might take a moment.")
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    
    print(f" Corpus updated. Total sentences now: {len(corpus)}")


def search(query: str, top_k: int = 3):
   
    if corpus_embeddings is None:
        print(" Corpus is empty. Please use the 'add' command to add text first.")
        return

    print(f"\n🔍 Searching for: '{query}'")
    
    
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    
    
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
   
    top_results = cosine_scores.argsort(descending=True)[:top_k]

    print("\n--- Top Results ---")
    for idx in top_results:
        print(f"- {corpus[idx]}  (Score: {cosine_scores[idx]:.4f})")
    print("-------------------\n")


if __name__ == "__main__":
    print("\n Welcome to the In-Memory Semantic Search Tool ")
    print("Commands: 'add' (to add text), 'search' (to find similar sentences), 'exit' (to quit).")

    while True:
        command = input("\nEnter command (add/search/exit): ").strip().lower()

        if command == "exit":
            print(" Exiting the tool. Goodbye!")
            break
        elif command == "add":
            paragraph = input("\n Paste the text you want to add to the corpus:\n")
            update_corpus(paragraph)
        elif command == "search":
            query = input("\n Enter your search query:\n")
            search(query)
        else:
            print(" Unknown command. Please use 'add', 'search', or 'exit'.")