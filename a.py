# -*- coding: utf-8 -*-
"""
Semantic Search Engine (Live Web via SerpApi)
"""

import gradio as gr
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
from serpapi import GoogleSearch
import torch
import nltk
import os

# Ensure NLTK tokenizer data is available
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Your SerpApi API key
SERPAPI_KEY = "8474514cda9ae18e8da9f83258a2f61fe09f148a713f3e2c64e483f688de14ab"


def get_articles(query, num_results=5):
    """Fetch articles from Google via SerpApi."""
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    docs = []
    for r in results.get("organic_results", []):
        url = r.get("link")
        if not url:
            continue
        try:
            article = Article(url)
            article.download()
            article.parse()
            if len(article.text) > 50:
                docs.append((url, article.text))
        except Exception as e:
            print(f"Error fetching article {url}: {e}")
            continue
    return docs


def semantic_search(query):
    """Fetch live articles, compute cosine similarity, and return top sentences."""
    docs = get_articles(query, num_results=5)
    if len(docs) == 0:
        return "No live articles found for this query."

    urls, texts = zip(*docs)
    corpus_embeddings = model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=min(3, len(texts)))

    output = []
    for score, idx in zip(top_results.values, top_results.indices):
        article_text = texts[idx]
        snippet = article_text[:300] + "..."

        sentences = sent_tokenize(article_text)
        sent_embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        sent_scores = util.cos_sim(query_embedding, sent_embeddings)[0]
        top_sents_idx = torch.topk(sent_scores, k=min(3, len(sentences)))
        top_sentences = [(float(sent_scores[i]), sentences[i]) for i in top_sents_idx.indices]

        out_text = f"**Article URL:** {urls[idx]}\n"
        out_text += f"**Article Cosine Similarity:** {float(score):.4f}\n"
        out_text += f"**Snippet:** {snippet}\n"
        out_text += "**Top 3 sentences by similarity:**\n"
        for s_score, s in top_sentences:
            out_text += f"  {s_score:.4f} | {s}\n"
        output.append(out_text)
    return "\n\n".join(output)


iface = gr.Interface(
    fn=semantic_search,
    inputs=gr.Textbox(lines=1, placeholder="Enter your query here..."),
    outputs=gr.Textbox(lines=20),
    title="🔍 Semantic Search Engine (Live Web via SerpApi)",
    description="Type any query, fetches live articles from Google using SerpApi, ranks them by cosine similarity, and shows top sentences."
)

if __name__ == "__main__":
    iface.launch(debug=True, share=True)
