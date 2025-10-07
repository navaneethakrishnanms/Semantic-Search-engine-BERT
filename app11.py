
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
from serpapi import GoogleSearch
import torch
import nltk
import os


print("Ensuring NLTK data packages are available...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' package...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt_tab' package...")
    nltk.download('punkt_tab', quiet=True)
print("NLTK check complete.")

from nltk.tokenize import sent_tokenize


SERPAPI_KEY = "8474514cda9ae18e8da9f83258a2f61fe09f148a713f3e2c64e483f688de14ab"



MODEL_NAME = 'all-MiniLM-L6-v2'
print(f"Loading sentence transformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")


def get_live_articles(query: str, num_results: int = 5) -> list[tuple[str, str]]:
    
    print(f"Fetching {num_results} articles for query: '{query}'")
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    docs = []
    organic_results = results.get("organic_results", [])
    
    for result in organic_results:
        url = result.get("link")
        if not url:
            continue
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 100:
                docs.append((url, article.text))
        except Exception as e:
            print(f"Skipping article from {url}. Reason: {e}")
            continue
            
    print(f"Successfully fetched and parsed {len(docs)} articles.")
    return docs


def semantic_search(query: str) -> str:
    """
    Performs a full semantic search pipeline.
    """
    articles = get_live_articles(query, num_results=5)
    
    if not articles:
        return "Could not find any live articles for this query. Please try a different query."

    urls, texts = zip(*articles)
    
    print("Encoding query...")
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    
    print("Encoding full articles...")
    article_embeddings = model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
    
    article_scores = util.cos_sim(query_embedding, article_embeddings)[0]
    
    top_article_results = torch.topk(article_scores, k=min(3, len(articles)))

    output_parts = []
    print("Processing top articles to find best sentences...")
    for i, (score, idx) in enumerate(zip(top_article_results.values, top_article_results.indices)):
        article_url = urls[idx]
        article_text = texts[idx]
        
        sentences = sent_tokenize(article_text)
        if not sentences:
            continue

        sent_embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        sent_scores = util.cos_sim(query_embedding, sent_embeddings)[0]
        top_sents_idx = torch.topk(sent_scores, k=min(3, len(sentences)))

        out_text = f"### Result {i+1}: Best match from [{article_url}]({article_url})\n"
        out_text += f"**Overall Article Similarity:** {score.item():.4f}\n\n"
        out_text += "**Top 3 Sentences:**\n"
        
        for sent_score, sent_idx in zip(top_sents_idx.values, top_sents_idx.indices):
            out_text += f"- (Score: {sent_score.item():.4f}) {sentences[sent_idx]}\n"
        
        output_parts.append(out_text)
        
    return "\n---\n".join(output_parts)



iface = gr.Interface(
    fn=semantic_search,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here... e.g., 'What are the benefits of transformer models in AI?'"),
    outputs=gr.Markdown(line_breaks=True),
    title="🔍 Live Web Semantic Search",
    description="This engine fetches live articles from Google, ranks them by semantic similarity, and then extracts the most relevant sentences from the top results. Powered by SerpApi, Sentence-Transformers, and Gradio.",
    examples=[
        ["latest advancements in battery technology"],
        ["health benefits of a mediterranean diet"],
        ["impact of remote work on productivity"]
    ]
)

if __name__ == "__main__":
    iface.launch(debug=True)