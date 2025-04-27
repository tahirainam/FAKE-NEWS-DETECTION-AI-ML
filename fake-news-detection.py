import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Load the fake and true datasets
fake_df = pd.read_csv('DataSet_Misinfo_FAKE.csv')
true_df = pd.read_csv('DataSet_Misinfo_TRUE.csv')

# Fake news label = 0
fake_df['label'] = 0

# Real news label = 1
true_df['label'] = 1


# Combine both into one dataset
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Check first few rows
print(combined_df.head())


print(combined_df.columns)




# Download stopwords (only first time)
nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define a clean_text function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)
    else:
        return ''

# Apply cleaning to the 'text' column
combined_df['cleaned_text'] = combined_df['text'].apply(clean_text)

# Check cleaned output
print(combined_df[['text', 'cleaned_text']].head())



from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned text
X = vectorizer.fit_transform(combined_df['cleaned_text']).toarray()

# Target labels
y = combined_df['label'].values

print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
