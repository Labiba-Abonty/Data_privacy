import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load the privacy policy from the text file
with open("bkash_privacy.txt", "r", encoding="utf-8") as file:
    privacy_policy = file.read()

# Load the app reviews from the CSV file
reviews_df = pd.read_csv("data/bkash_review.csv")
app_reviews = reviews_df["content"].tolist()

# Extract relevant privacy-related keywords from the privacy policy
privacy_keywords = ["personal data", "location tracking", "user information"]

# Formulate the queries
privacy_queries = ["user data privacy", "location tracking concerns", "data collection"]

# Tokenize and preprocess the privacy policy and app reviews
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([token for token in tokens if token.isalpha() and token not in stop_words])

privacy_policy = preprocess(privacy_policy)
app_reviews = [preprocess(review) for review in app_reviews]

# Use TF-IDF vectorizer to transform reviews into numerical vectors
vectorizer = TfidfVectorizer()
review_vectors = vectorizer.fit_transform(app_reviews)

# Transform the privacy queries into numerical vectors
privacy_query_vectors = vectorizer.transform(privacy_queries)

# Calculate cosine similarity between each review and the privacy queries
similarities = cosine_similarity(review_vectors, privacy_query_vectors)

# Rank the reviews based on their relevance to privacy concerns
# The index of the review in 'similarities' corresponds to the index in 'app_reviews'
sorted_reviews_indices = similarities.argsort(axis=0)[:, -1]
privacy_related_reviews = [app_reviews[i] for i in sorted_reviews_indices]

# Manually annotate the review candidates to create a labeled dataset
# For this example, assume that index 2, 3, 5, 7 are privacy-related reviews, and others are not.
labeled_dataset = {
    "positive_examples": [app_reviews[2], app_reviews[3], app_reviews[5], app_reviews[7]],
    "negative_examples": [app_reviews[i] for i in range(len(app_reviews)) if i not in [2, 3, 5, 7]],
}

# Balance the dataset (Optional)
# You can use techniques like random under-sampling or over-sampling if needed.

# Split the dataset into training and testing sets
X = labeled_dataset["positive_examples"] + labeled_dataset["negative_examples"]
y = [1] * len(labeled_dataset["positive_examples"]) + [0] * len(labeled_dataset["negative_examples"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you have the labeled dataset and can proceed to Step 2: Training the review classifier
# using X_train, y_train and evaluating its performance on X_test, y_test.
