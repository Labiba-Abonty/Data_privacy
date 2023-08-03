import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Read the preprocessed privacy policy text
with open('preprocessed_privacy_policy.txt', 'r', encoding='utf-8') as file:
    privacy_policy_text = file.read()

# Preprocess the text
preprocessed_text = preprocess_text(privacy_policy_text)

# Create a TF-IDF vectorizer with min_df=1 and remove stopwords
stop_words = stopwords.words('english')
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words=stop_words)

# Fit and transform the vectorizer on the preprocessed text
tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])

# Get feature names (words) from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get the TF-IDF scores for each word
tfidf_scores = tfidf_matrix.toarray()[0]

# Combine words with their TF-IDF scores into a dictionary
word_tfidf_scores = dict(zip(feature_names, tfidf_scores))

# Sort the dictionary based on TF-IDF scores in descending order
sorted_word_tfidf_scores = {k: v for k, v in sorted(word_tfidf_scores.items(), key=lambda item: item[1], reverse=True)}

# Extract the top 50 keywords related to privacy (excluding common stopwords)
stop_words = set(stopwords.words('english'))
top_keywords = [word for word in sorted_word_tfidf_scores.keys() if word not in stop_words][:50]

print(top_keywords)
