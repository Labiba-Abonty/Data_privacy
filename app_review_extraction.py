import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the privacy-related keywords
privacy_keywords = ['information', 'platform', 'nagad', 'agreement', 'cookies', 'including', 'user', 'third', 'party', 'access', 'content', 'account', 'rights', 'personal', 'services', 'service', 'terms', 'without', 'time', 'users', 'website', 'parties', 'data', 'used', 'also', 'bangladesh', 'right', 'send', 'applicable', 'property', 'andor', 'breach', 'business', 'device', 'purpose', 'transaction', 'computer', 'conditions', 'ideas', 'laws', 'material', 'otherwise', 'provide', 'provided', 'refund', 'request', 'responsible', 'advertising', 'merchant', 'mobile']

# Step 2: Load app reviews from the Excel file
def load_reviews(file_path):
    df = pd.read_excel(file_path)
    return df['review_description'].tolist()

file_path = 'data/nagad.csv'
app_reviews = load_reviews(file_path)

# Step 3: Convert all elements in the app_reviews list to strings
app_reviews = [str(review) for review in app_reviews]

# Step 4: Calculate relevance score using TF-IDF and Cosine Similarity
def calculate_relevance_score(query, reviews):
    tfidf_vectorizer = TfidfVectorizer(vocabulary=query)
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    similarity_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_scores.diagonal()

relevance_scores = calculate_relevance_score(privacy_keywords, app_reviews)

# Step 5: Rank the reviews based on their relevance scores
review_rankings = sorted(zip(relevance_scores, app_reviews), reverse=True)

# Step 6: Extract 1,000 privacy-related reviews
num_desired_reviews = 1000
privacy_related_reviews = []
for rank, review in review_rankings:
    if len(privacy_related_reviews) >= num_desired_reviews:
        break
    privacy_related_reviews.append(review)

# Step 7: Print the extracted privacy-related reviews
for review in privacy_related_reviews:
    print(f"Privacy-Related Review: {review}\n")
