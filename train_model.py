import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load real and fake datasets
real = pd.read_csv('real.csv')
fake = pd.read_csv('fake.csv')

# Add labels
real['label'] = 1
fake['label'] = 0

# Combine datasets
data = pd.concat([real, fake], ignore_index=True)
data = data[['title', 'label']].dropna()

# Preprocess: lowercase
data['title'] = data['title'].apply(lambda x: x.lower())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['title'], data['label'], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
os.makedirs("trained_models", exist_ok=True)
joblib.dump(model, 'trained_models/logistic_regression_model.joblib')
joblib.dump(vectorizer, 'trained_models/vectorizer.joblib')

print("✅ Model and vectorizer saved to 'trained_models/'")
