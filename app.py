import gradio as gr
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt', quiet=True)

model = SentenceTransformer("intfloat/e5-large-v2")


corpus = []
corpus_embeddings = None


def update_corpus(paragraph: str):
    global corpus, corpus_embeddings
    new_sentences = sent_tokenize(paragraph)
    if not new_sentences:
        return "No new sentences found."

    corpus.extend(new_sentences)

    prefixed_corpus = [f"passage: {sent}" for sent in corpus]

    corpus_embeddings = model.encode(
        prefixed_corpus, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    return f" Corpus updated. Total sentences now: {len(corpus)}"

def search(query: str, top_k: int = 3):
    if corpus_embeddings is None:
        return " Corpus is empty. Add text first."

    
    query = f"query: {query}"

    query_embedding = model.encode(
        query, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = cosine_scores.argsort(descending=True)[:top_k]
    results = [(corpus[idx], float(cosine_scores[idx])) for idx in top_results]
    return results

def add_text_ui(paragraph):
    return update_corpus(paragraph)

def search_ui(query, top_k):
    results = search(query, top_k)
    if isinstance(results, str):
        return results
    return "\n\n".join([
        f"{i+1}. {sent}\n(Score: {score:.4f})"
        for i, (sent, score) in enumerate(results)
    ])


demo = gr.Interface(
    fn=lambda paragraph, query, top_k: (
        add_text_ui(paragraph), 
        search_ui(query, int(top_k))
    ),
    inputs=[
        gr.Textbox(lines=8, label="📝 Add Text to Corpus", placeholder="Paste your paragraph here..."),
        gr.Textbox(label="🔍 Search Query", placeholder="Enter your search phrase..."),
        gr.Slider(1, 10, step=1, value=3, label="Top K Results")
    ],
    outputs=[
        gr.Textbox(label="📚 Corpus Update Status", lines=2, interactive=False),
        gr.Textbox(label="🎯 Search Results", lines=15, interactive=False)
    ],
    title="🔎 Advanced Semantic Search (E5-Large-v2)",
    description="Add paragraphs, and search semantically similar sentences using a high-accuracy transformer model.",
)


demo.launch(share=False)
