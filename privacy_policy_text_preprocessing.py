import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    additional_stop_words = {'shall', 'may', 'use', 'us'}
    stop_words = stop_words.union(additional_stop_words)
    words = [word for word in words if word not in stop_words]

    # Join the words back to a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

# Read the privacy policy text
with open('privacy_policy.txt', 'r', encoding='utf-8') as file:
    privacy_policy_text = file.read()

# Preprocess the text
preprocessed_text = preprocess_text(privacy_policy_text)

# Save the preprocessed text to a new file
with open('preprocessed_privacy_policy.txt', 'w', encoding='utf-8') as file:
    file.write(preprocessed_text)
