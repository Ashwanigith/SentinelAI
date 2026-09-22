import os
import re
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Preprocess text: lowercasing, regex cleaning, and stopword removal."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\d+', ' numtoken ', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Training dataset
data = {
    'text': [
        'Hey, are we still meeting for lunch at 1 PM today?',
        'Can you send me the report by end of day please?',
        'Sure, let me check the documents and get back to you.',
        'What time does the train leave tomorrow morning?',
        'Happy birthday! Hope you have a wonderful day ahead.',
        'Please review the attached PR when you get a chance.',
        'URGENT: Your account has been suspended! Verify your password at http://bit.ly/fake-bank immediately.',
        'Congratulations! You won a $1,000 Walmart gift card. Click here to claim your prize now!',
        'FINAL WARNING: Your debit card is blocked. Send OTP to unlock access within 10 minutes.',
        'Work from home and earn $500 daily without investment. Register now at link!',
        'SECURITY ALERT: Unauthorized login detected from Russia. Click here to secure your account immediately.',
        'Exclusive deal: 90% off all designer watches today only! Claim voucher before it expires.'
    ],
    'label': ['Ham', 'Ham', 'Ham', 'Ham', 'Ham', 'Ham', 'Threat/Spam', 'Threat/Spam', 'Threat/Spam', 'Threat/Spam', 'Threat/Spam', 'Threat/Spam']
}

df = pd.DataFrame(data)
df['cleaned'] = df['text'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds):.2f}")

# Save artifacts
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/spam_classifier.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
print("Model and vectorizer successfully saved in the 'model/' directory.")