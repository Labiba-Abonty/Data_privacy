import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the labeled dataset
file_path = 'labiba158.xlsx'
df = pd.read_excel(file_path)

# Step 2: Preprocess the text data
def preprocess_text(text):
    text = str(text).lower()
    # You can add more text preprocessing steps if required
    return text

df['preprocessed_review'] = df['Review'].apply(preprocess_text)

# Step 3: Split the dataset into training and testing sets
X = df['preprocessed_review']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Transform text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 5: Fine-tune the SVM classifier
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

classifier = SVC()
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

best_classifier = grid_search.best_estimator_

# Step 6: Make predictions on the test set
y_pred = best_classifier.predict(X_test_tfidf)

# Step 7: Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
