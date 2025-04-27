import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load nltk stuff
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load your trained model and vectorizer
model = LogisticRegression(max_iter=1000)
vectorizer = TfidfVectorizer(max_features=5000)

# Reload training part again to fit vectorizer and model
import pandas as pd

# Load dataset
fake_df = pd.read_csv('DataSet_Misinfo_FAKE.csv')
true_df = pd.read_csv('DataSet_Misinfo_TRUE.csv')

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine
combined_df = pd.concat([fake_df, true_df])

# Clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)
    else:
        return ''

combined_df['cleaned_text'] = combined_df['text'].apply(clean_text)

# Fit vectorizer and model again
X = vectorizer.fit_transform(combined_df['cleaned_text']).toarray()
y = combined_df['label'].values
model.fit(X, y)

# Streamlit App
st.title("📰 Fake News Detection App")
st.write("Enter a news article below:")

user_input = st.text_area("News Article")

if st.button("Detect"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    
    if prediction == 0:
        st.error("🔴 This news is likely **FAKE**.")
    else:
        st.success("🟢 This news is likely **REAL**.")
