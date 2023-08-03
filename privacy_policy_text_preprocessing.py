import nltk
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from cltk.tokenize.sentence import TokenizeSentence

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def lower_case(text):
    return text.lower()

def no_digit(text):
    return ''.join([i for i in text if not i.isdigit()])

def no_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def no_whitespace(text):
    return text.strip()

def tokenize(text):
    return word_tokenize(text)

def stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [i for i in tokens if not i in stop_words]

def stemming(text):
    stemmer = PorterStemmer()
    input_str = word_tokenize(text)
    return [stemmer.stem(word) for word in input_str]

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    input_str = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in input_str]

def pos(text):
    result = TextBlob(text)
    return result.tags

def chunking(text):
    result = TextBlob(text)
    reg_exp = "NP: {<DT>?<JJ>*<NN>}"
    rp = nltk.RegexpParser(reg_exp)
    return rp.parse(result.tags)

def named_entity(text):
    entity = ne_chunk(pos_tag(word_tokenize(text)))
    return entity

def process_text_file(input_file_path, preprocessing_functions):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    for func in preprocessing_functions:
        text = func(text)

    return text

def save_preprocessed_text(output_file_path, preprocessed_text):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(preprocessed_text)

# Preprocessing functions for English text
english_preprocessing_functions = [lower_case, no_digit, no_punctuation, no_whitespace, tokenize, stop_words, stemming, lemmatization]

# Preprocessing functions for Bengali text
bengali_preprocessing_functions = [no_digit, no_punctuation, no_whitespace, bangla_tokenize]

# Process English text
english_input_file = 'privacy_policy_nexus_en.txt'
english_output_file = 'privacy_policy_nexus_en_preprocessed.txt'
english_preprocessed_text = process_text_file(english_input_file, english_preprocessing_functions)
save_preprocessed_text(english_output_file, english_preprocessed_text)

# Process Bengali text
bengali_input_file = 'privacy_policy_nexus.bn.txt'
bengali_output_file = 'privacy_policy_nexus_bn_preprocessed.txt'
bengali_preprocessed_text = process_text_file(bengali_input_file, bengali_preprocessing_functions)
save_preprocessed_text(bengali_output_file, bengali_preprocessed_text)
