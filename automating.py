import pandas as pd
import spacy

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Define the security/privacy-related base words
base_words = ["auth", "secure", "protect", "confidential","track","privacy"]

def get_lemmas(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc]

def label_reviews(excel_file_path, output_file_path):
    # Read the reviews from the Excel file
    df = pd.read_excel(excel_file_path)

    # Create an empty list to store the labeled reviews
    labeled_reviews = []

    for review in df['review_description']:
        # Process the review and get its lemmatized tokens
        lemmatized_tokens = get_lemmas(review)

        # Check if any of the base words or their forms are present in the review
        is_security_related = any(base_word in lemmatized_tokens for base_word in base_words)

        # Label the review based on the presence of keywords
        label = 1 if is_security_related else 0

        # Append the review and its label to the list
        labeled_reviews.append((review, label))

    # Create a new DataFrame for the labeled reviews
    labeled_df = pd.DataFrame(labeled_reviews, columns=['review_description', 'label'])

    # Save the labeled data to a new Excel file
    labeled_df.to_excel(output_file_path, index=False, engine='openpyxl')
    print("Review labeling process completed.")

if __name__ == "__main__":
    input_excel_file = "nagad1.xlsx"
    output_excel_file = "labeled_nagad.xlsx"

    label_reviews(input_excel_file, output_excel_file)
