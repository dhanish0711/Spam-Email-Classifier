import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
COMBINED_DATA_PATH = os.path.join(DATA_DIR, "combined_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")

# Dataset URLs and local filenames
DATASETS = {
    "enron": {
        "url": "https://raw.githubusercontent.com/ajaykuma/Datasets_For_Work/master/spam_ham_dataset.csv",
        "filename": "spam_ham_dataset.csv"
    },
    "spamassassin": {
        "url": "https://raw.githubusercontent.com/chloeoxe/spam-email-detector/master/completeSpamAssassin.csv",
        "filename": "completeSpamAssassin.csv"
    },
    "sms_spam": {
        "url": "https://raw.githubusercontent.com/justmarkham/DAT5/master/data/SMSSpamCollection.txt",
        "filename": "SMSSpamCollection"
    },
    "kaggle_spam": {
        "url": "https://raw.githubusercontent.com/campusx-official/sms-spam-classifier/main/spam.csv",
        "filename": "spam.csv"
    }
}

def ensure_dirs():
    """Ensure data and models directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def download_datasets():
    """Download each dataset if not already present locally."""
    ensure_dirs()
    print("=== Downloading Datasets ===")
    for name, info in DATASETS.items():
        local_path = os.path.join(DATA_DIR, info["filename"])
        if not os.path.exists(local_path):
            print(f"Downloading {name} dataset from {info['url']}...")
            try:
                req = urllib.request.Request(
                    info["url"],
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"  Successfully downloaded and saved to {local_path}.")
            except Exception as e:
                print(f"  Error downloading {name} dataset: {e}")
        else:
            print(f"{name} dataset already exists locally at {local_path}.")

def load_and_preprocess():
    """Load all 4 datasets, standardize columns, merge, and clean them."""
    dfs = []
    
    # 1. Enron Spam Subset
    enron_path = os.path.join(DATA_DIR, DATASETS["enron"]["filename"])
    if os.path.exists(enron_path):
        print("\nProcessing Enron Spam dataset...")
        try:
            df = pd.read_csv(enron_path)
            df = df[['text', 'label']].copy()
            df['label'] = df['label'].map({'spam': 'Spam', 'ham': 'Not Spam'})
            df = df.dropna()
            print(f"  Loaded {len(df)} samples.")
            dfs.append(df)
        except Exception as e:
            print(f"  Error processing Enron Spam dataset: {e}")

    # 2. SpamAssassin Subset
    sa_path = os.path.join(DATA_DIR, DATASETS["spamassassin"]["filename"])
    if os.path.exists(sa_path):
        print("Processing SpamAssassin dataset...")
        try:
            df = pd.read_csv(sa_path)
            df = df[['Body', 'Label']].copy()
            df.rename(columns={'Body': 'text', 'Label': 'label'}, inplace=True)
            df['label'] = df['label'].map({1: 'Spam', 0: 'Not Spam'})
            df = df.dropna()
            print(f"  Loaded {len(df)} samples.")
            dfs.append(df)
        except Exception as e:
            print(f"  Error processing SpamAssassin dataset: {e}")

    # 3. SMS Spam Collection
    sms_path = os.path.join(DATA_DIR, DATASETS["sms_spam"]["filename"])
    if os.path.exists(sms_path):
        print("Processing SMS Spam Collection...")
        try:
            df = pd.read_csv(sms_path, sep='\t', header=None, names=['label', 'text'])
            df['label'] = df['label'].map({'spam': 'Spam', 'ham': 'Not Spam'})
            df = df.dropna()
            print(f"  Loaded {len(df)} samples.")
            dfs.append(df)
        except Exception as e:
            print(f"  Error processing SMS Spam Collection: {e}")

    # 4. Kaggle Spam/SMS CSV
    kaggle_path = os.path.join(DATA_DIR, DATASETS["kaggle_spam"]["filename"])
    if os.path.exists(kaggle_path):
        print("Processing Kaggle Spam dataset...")
        try:
            df = pd.read_csv(kaggle_path, encoding='latin-1')
            df = df[['v1', 'v2']].copy()
            df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
            df['label'] = df['label'].map({'spam': 'Spam', 'ham': 'Not Spam'})
            df = df.dropna()
            print(f"  Loaded {len(df)} samples.")
            dfs.append(df)
        except Exception as e:
            print(f"  Error processing Kaggle Spam dataset: {e}")

    if not dfs:
        raise ValueError("No datasets could be loaded successfully. Please check the downloads.")

    # Combine all
    print("\nMerging datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    initial_len = len(combined_df)
    
    # Data cleaning
    combined_df['text'] = combined_df['text'].astype(str).str.strip()
    combined_df = combined_df[combined_df['text'] != '']
    combined_df = combined_df.dropna(subset=['text', 'label'])
    combined_df = combined_df.drop_duplicates(subset=['text'])
    
    final_len = len(combined_df)
    print(f"Merged row count: {initial_len}")
    print(f"After deduplication and cleaning: {final_len} samples.")
    
    combined_df.to_csv(COMBINED_DATA_PATH, index=False)
    print(f"Combined dataset saved to {COMBINED_DATA_PATH}")
    
    distribution = combined_df['label'].value_counts()
    print("\nClass Distribution:")
    for cls, count in distribution.items():
        percentage = (count / final_len) * 100
        print(f"  {cls}: {count} ({percentage:.2f}%)")
        
    return combined_df

def train_and_evaluate(df):
    """Train the model and print evaluation metrics."""
    print("\n=== Model Training and Evaluation ===")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, min_df=2)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["Not Spam", "Spam"])
    print(f"               Predicted Not Spam   Predicted Spam")
    print(f"Actual Not Spam      {cm[0][0]:<15}      {cm[0][1]:<15}")
    print(f"Actual Spam          {cm[1][0]:<15}      {cm[1][1]:<15}")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

if __name__ == "__main__":
    download_datasets()
    combined_df = load_and_preprocess()
    train_and_evaluate(combined_df)
