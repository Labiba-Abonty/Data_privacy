import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Preprocessing and Feature Extraction
df = pd.read_excel("labiba158.xlsx")


# Assuming you have already preprocessed the text and saved it in the 'preprocessed_review' column
X = df['Review'].tolist()
y = df['Label'].tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Random Forest Classifier
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train_tfidf, y_train)

# Step 3: Evaluate the Random Forest Classifier
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred = random_forest_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
