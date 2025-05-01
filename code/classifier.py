import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


# Step 1: Load data
df = pd.read_csv("../data/6_sorted_quoted.csv")

# Load sentence embeddings (if not already saved in DataFrame)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_message(msg):
    return [word.strip().lower() for word in str(msg).split(',') if word.strip()]

df['tokens'] = df['message'].apply(preprocess_message)
df['clean_sentence'] = df['tokens'].apply(lambda words: ' '.join(words))
df['embedding'] = df['clean_sentence'].apply(lambda x: model.encode(x))

# Convert embeddings and labels
X = np.vstack(df['embedding'].to_numpy())
y = df['label'].apply(lambda x: 1 if x.strip().lower() == 'fraud' else 0).to_numpy()

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train classifier
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate classifier
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['normal', 'fraud'])

print("📊 Classification Report:\n")
print(report)

# Step 5: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'fraud'])

# Create output directory
os.makedirs("../results/classification_report", exist_ok=True)

# Save confusion matrix
plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Logistic Regression)")
plt.savefig("../results/classification_report/confusion_matrix.png", bbox_inches='tight')
plt.close()

# Step 6: Save classification report as text
with open("../results/classification_report/classification_report.txt", "w") as f:
    f.write(report)

# Optional: Save prediction vs actual as CSV
df_test = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
df_test.to_csv("../results/classification_report/test_predictions.csv", index=False)

# If possible, extract feature importances from the model
# For logistic regression, we can use the coefficients as a proxy for feature importance
if hasattr(clf, 'coef_'):
    # Get feature importances and map to words
    importances = np.abs(clf.coef_[0])
    
    # Create a mapping from embedding dimensions to importance
    feature_importance = pd.DataFrame({
        'feature': range(len(importances)),
        'importance': importances
    })
    
    # Optional: If you have the vocabulary from your embedding model
    # Map the embeddings back to words - this is a simplified approach
    # You'd need to adapt this based on your specific embedding approach
    try:
        # Get a sample of the most used words in the dataset
        from collections import Counter
        all_words = [word for words in df['tokens'] for word in words]
        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(100)]
        
        # Get embeddings for these words
        word_embeddings = {word: model.encode(word) for word in top_words}
        
        # Calculate correlation between each word embedding and model coefficients
        word_importance = []
        for word, embedding in word_embeddings.items():
            # Calculate a simplified importance score based on dot product
            importance = np.abs(np.dot(embedding, clf.coef_[0]))
            word_importance.append({'word': word, 'importance': importance})
        
        # Create DataFrame and save
        word_imp_df = pd.DataFrame(word_importance)
        word_imp_df.sort_values('importance', ascending=False, inplace=True)
        word_imp_df.to_csv("../results/classification_report/word_importances.csv", index=False)
    except Exception as e:
        print(f"Could not map feature importances to words: {e}")




print("✅ Model trained and results saved to ../results/classification_report")

joblib.dump(clf, "../results/classification_report/logistic_model.joblib")
