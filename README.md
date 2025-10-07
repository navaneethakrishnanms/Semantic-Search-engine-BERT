🔎 Advanced Semantic Search (E5-Large-v2)
🚀 Search for Meaning — Not Just Keywords!

This project is an interactive semantic search web app built using Gradio and Sentence Transformers.
It allows you to add paragraphs of text, automatically breaks them into sentences, and enables semantic similarity search using the E5-Large-v2 embedding model.

## 🌐 Live Demo  
👉 [**Launch the App on Hugging Face 🚀**](https://huggingface.co/spaces/navaneethakrishnan1234/BERT-Semantic-search-engine)


👉 Launch the App

(Replace the above link with your actual app link if you share it via Gradio or Hugging Face Spaces.)

✨ Features

✅ Semantic Understanding — Finds sentences with similar meaning, not just matching keywords.
✅ E5-Large-v2 Model — High-performance, multilingual embedding model from Hugging Face.
✅ Interactive Web UI — Built with Gradio
 for easy use.
✅ Sentence Tokenization — Automatically splits your input paragraph into searchable sentences.
✅ Dynamic Corpus — Add text anytime and search instantly.
✅ Top-K Results — Customize how many similar results you want to view.

🧠 How It Works

You input a paragraph into the app.

The app uses NLTK to split it into sentences.

Each sentence is encoded into a semantic embedding vector using intfloat/e5-large-v2.

During a search, your query is also converted into an embedding.

The app computes cosine similarity between the query and all stored sentences.

The top K most semantically similar sentences are displayed with similarity scores.

🧩 Tech Stack
Component	Description
🧠 Model	SentenceTransformer - intfloat/e5-large-v2

🧰 Framework	Gradio
 — for interactive UI
🔤 NLP Tool	NLTK
 — for sentence tokenization
🐍 Language	Python 3.8+
🧮 Math	Cosine Similarity (via sentence_transformers.util)
⚙️ Installation & Setup
1️⃣ Clone this Repository
git clone https://github.com/your-username/semantic-search-e5-large-v2.git
cd semantic-search-e5-large-v2

2️⃣ Install Dependencies
pip install gradio sentence-transformers nltk

3️⃣ Run the App
python app.py


By default, the app runs locally at:
👉 http://127.0.0.1:7860/

💡 Usage

Paste a paragraph into the 📝 Add Text to Corpus box.

Enter your 🔍 search query (e.g., “machine learning applications”).

Adjust Top K Results slider to choose how many top matches to show.

Hit Submit — the app returns the most semantically similar sentences with similarity scores.

🧾 Example
Input Paragraph:

Artificial intelligence is transforming industries.
Machine learning models are trained on large datasets.
Deep learning is a subset of machine learning that uses neural networks.

Search Query:

How does deep learning work?

Output:
1. Deep learning is a subset of machine learning that uses neural networks.
(Score: 0.91)

2. Machine learning models are trained on large datasets.
(Score: 0.74)

📸 UI Preview
Add Text	Search Results

	

(Replace with real screenshots of your app if available)

🧩 Folder Structure
semantic-search-e5-large-v2/
│
├── app.py               # Main application code (Gradio + E5 model)
├── requirements.txt     # Dependencies list
├── README.md            # Documentation
└── assets/              # (Optional) Screenshots or app images

🔍 Key Functions Explained
Function	Description
update_corpus(paragraph)	Splits input paragraph into sentences, encodes them, and adds to the corpus.
search(query, top_k)	Finds the most semantically similar sentences using cosine similarity.
reparameterize()	(Not used here) — but analogous concept in VAEs for sampling latent space.
demo.launch()	Launches the Gradio app locally or shares it publicly.
🚀 Future Enhancements

 Add vector database integration (like FAISS or ChromaDB).

 Enable document-level search.

 Add visualization of embeddings (2D projection with PCA/UMAP).

 Deploy to Hugging Face Spaces or Streamlit Cloud.

🧑‍💻 Author

[NAVANEETHA KRISHNAN M S]