from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-MiniLM-L6-v2')


corpus = [
    "Apple Inc. announced a record quarterly revenue driven by iPhone sales.",
    "Apple released macOS update with security patches.",
    "Apple unveiled the new iPhone 17 with an improved camera system.",
    "Apple's services revenue grew year-over-year thanks to App Store subscriptions.",
    "Apple is reportedly exploring mixed reality headset prototypes.",
    "Tim Cook discussed supply chain challenges during the earnings call.",
    "Apple's stock climbed after the company announced a major buyback program.",
    "Developers praised the new Xcode features announced at WWDC.",
    "Analysts expect Apple to expand its electric vehicle project.",
    "Apple Maps introduced transit updates for several cities.",
    "Apple faces regulatory scrutiny over App Store commission policies.",
    "Apple partnered with a health startup to integrate new health features.",
    "The MacBook Air now ships with an M3 chip and better battery life.",
    "Apple refreshed its iPad lineup with a faster processor.",
    "Apple announced a trade-in program for older devices.",
    "I picked a ripe red apple from the tree in the orchard.",
    "Apple pie is a classic dessert served with a scoop of vanilla ice cream.",
    "Granny Smith apples are tart and great for baking.",
    "A medium apple contains dietary fiber and vitamin C.",
    "The fruit market sells apples, pears, and seasonal berries.",
    "Slicing apples thinly helps them caramelize faster in a pan.",
    "Homemade apple sauce is simple: cook apples with cinnamon and sugar.",
    "An apple a day keeps the doctor away is an old proverb.",
    "The apple orchard had rows of trees stretching to the horizon.",
    "Cider makers press fresh apples to make hard cider.",
    "Dried apple chips make a healthy snack for hiking.",
    "The recipe calls for two cups of peeled and chopped apples.",
    "The apple tree outside my window blooms in spring with fragrant flowers.",
    "He used Apple Pay to complete the contactless payment at the cafe.",
    "Students use the Apple Pencil for digital art on the iPad Pro.",
    "Apple's encryption policies were discussed in a privacy forum.",
    "She placed an apple on the teacher's desk as a gift.",
    "A local bakery featured apple cinnamon rolls this weekend.",
    "The classroom experiment measured how apples fall from the tree.",
    "Python is a popular programming language for machine learning.",
    "The Eiffel Tower attracts millions of tourists every year.",
    "Solar panels reduce electricity bills by generating renewable energy.",
    "Vaccination campaigns helped control the spread of infectious diseases.",
    "The stock market rallied after positive economic data was released.",
    "Researchers published a paper on neural network pruning techniques.",
    "Electric vehicles are becoming more affordable due to battery advances.",
    "The movie festival showcased independent films from around the world.",
    "Farmers rotated crops to improve soil health and yields.",
    "The climate conference focused on cutting greenhouse gas emissions.",
    "Apple launched a subscription bundle combining Apple TV+, Music, and Arcade.",
    "The company expanded its retail presence with new stores in Asia.",
    "Apple's repair program now offers same-day screen replacements.",
    "Supplier reports indicate improved component availability for Apple.",
    "Apple updated its developer guidelines to support new APIs.",
    "The keynote emphasized Apple's commitment to privacy and security.",
    "New enterprise features make Apple devices easier to manage at scale.",
    "Apple collaborated with universities on carbon-neutral manufacturing techniques."
]


corpus_embeddings = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

def search(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = cosine_scores.argsort(descending=True)[:top_k]

    print(f"\n Query: {query}\n")
    for idx in top_results:
        print(f"{corpus[idx]}  (Score: {cosine_scores[idx]:.4f})")


if __name__ == "__main__":
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        search(query)
