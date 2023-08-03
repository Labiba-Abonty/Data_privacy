import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the annotated privacy-related reviews
file_path = 'labiba158.xlsx'
df = pd.read_excel(file_path)

# Step 2: Preprocess the reviews
def preprocess_text(text):
    # Add your text preprocessing code here (e.g., lowercase, remove punctuation, remove stop words)
    preprocessed_text = text.lower()  # Replace this with your actual preprocessing logic
    return preprocessed_text

# Preprocess the "Review" column
df['preprocessed_review'] = df['Review'].apply(preprocess_text)

# Step 3: Split the dataset into training and testing sets
X = df['preprocessed_review']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a TF-IDF vectorizer and transform the training data
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Step 5: Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Step 6: Transform the testing data and make predictions
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)

# Step 7: Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)
