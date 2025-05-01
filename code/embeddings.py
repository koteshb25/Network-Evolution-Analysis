import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import defaultdict
import umap

# Setup stopwords
stop_words = set(stopwords.words('english'))

# Step 1: Load & Clean Dataset
df = pd.read_csv("../data/6_sorted_quoted.csv")

def preprocess_message(msg):
    return [word.strip().lower() for word in str(msg).split(',') if word.strip()]

df['tokens'] = df['message'].apply(preprocess_message)
df['tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
df['clean_sentence'] = df['tokens'].apply(lambda words: ' '.join(words))
sentences = df['clean_sentence'].tolist()

# Step 2: Sentence Embeddings using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences, show_progress_bar=True)
df['sentence_embedding'] = list(sentence_embeddings)

# Step 3: Visualize Sentence Embeddings (UMAP)
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine')
sentence_2d = reducer.fit_transform(sentence_embeddings)

# Step 4: Clustering (KMeans & silhouette check)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(sentence_embeddings)
df['sentence_cluster'] = clusters
sil_score = silhouette_score(sentence_embeddings, clusters)
print(f"Silhouette Score (KMeans): {sil_score:.3f}")

# Step 5: Word Embeddings using Word2Vec
word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=3, workers=4)
word_vectors = word2vec_model.wv
top_words = [word for word in word_vectors.index_to_key[:600]]
word_vecs = [word_vectors[word] for word in top_words]

# NEW: Count how often each word appears in fraud vs normal
word_label_counts = defaultdict(lambda: {'fraud': 0, 'normal': 0})
for tokens, label in zip(df['tokens'], df['label']):
    label = label.strip().lower()
    for word in set(tokens):
        word_label_counts[word][label] += 1

# NEW: Determine dominant label for each word
def dominant_label(word):
    counts = word_label_counts[word]
    return 'fraud' if counts['fraud'] > counts['normal'] else 'normal'

word_labels = [dominant_label(word) for word in top_words]
palette = {'fraud': 'red', 'normal': 'blue'}

# Step 6: UMAP for word embeddings + color by label
word_2d = reducer.fit_transform(word_vecs)

plt.figure(figsize=(14, 10))
for i, word in enumerate(top_words):
    x, y = word_2d[i]
    color = palette[word_labels[i]]
    plt.scatter(x, y, color=color)
    plt.text(x + 0.01, y + 0.01, word, fontsize=9, color=color)
plt.title("Word2Vec Embeddings Colored by Dominant Usage (Fraud vs Normal)")
plt.grid(True)
plt.savefig("../results/word2vec_by_label.png", dpi=300, bbox_inches='tight')

# Step 7: Plot Sentence Embeddings
plt.figure(figsize=(12, 8))
sns.scatterplot(x=sentence_2d[:, 0], y=sentence_2d[:, 1], hue=df['label'], palette='Set2')
plt.title("Sentence Embeddings (UMAP Projection)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.legend(title="Label")
plt.savefig("../results/sentence_clusters.png", dpi=300, bbox_inches='tight')
