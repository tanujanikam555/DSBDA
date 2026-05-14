 ----------------------------------------------------------
# STEP 1: Import Libraries
# ----------------------------------------------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download (first time only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ----------------------------------------------------------
# STEP 2: Sample Documents (2 documents)
# ----------------------------------------------------------
documents = [
    "Natural Language Processing is a fascinating field of Artificial Intelligence.",
    "Artificial Intelligence helps computers understand human language."
]

print("\nOriginal Documents:")
for i, doc in enumerate(documents, 1):
    print(f"Doc {i}: {doc}")

# ----------------------------------------------------------
# STEP 3: Tokenization
# ----------------------------------------------------------
print("\nTokenization:")
all_tokens = []
for i, doc in enumerate(documents, 1):
    tokens = word_tokenize(doc)
    all_tokens.append(tokens)
    print(f"Doc {i} Tokens:", tokens)

# ----------------------------------------------------------
# STEP 4: POS Tagging
# ----------------------------------------------------------
print("\nPOS Tagging:")
for i, tokens in enumerate(all_tokens, 1):
    pos_tags = pos_tag(tokens)
    print(f"Doc {i} POS Tags:", pos_tags)

# ----------------------------------------------------------
# STEP 5: Stop Words Removal
# ----------------------------------------------------------
stop_words = set(stopwords.words('english'))
filtered_docs = []

print("\nAfter Stop Words Removal:")
for i, tokens in enumerate(all_tokens, 1):
    filtered = [word for word in tokens if word.lower() not in stop_words]
    filtered_docs.append(filtered)
    print(f"Doc {i}:", filtered)

# ----------------------------------------------------------
# STEP 6: Stemming
# ----------------------------------------------------------
ps = PorterStemmer()
print("\nStemming:")
for i, tokens in enumerate(filtered_docs, 1):
    stemmed = [ps.stem(word) for word in tokens]
    print(f"Doc {i}:", stemmed)

# ----------------------------------------------------------
# STEP 7: Lemmatization
# ----------------------------------------------------------
lemmatizer = WordNetLemmatizer()
print("\nLemmatization:")
for i, tokens in enumerate(filtered_docs, 1):
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    print(f"Doc {i}:", lemmatized)

# ----------------------------------------------------------
# STEP 8: Term Frequency (CountVectorizer)
# ----------------------------------------------------------
cv = CountVectorizer()
tf_matrix = cv.fit_transform(documents)

print("\nTerm Frequency (CountVectorizer):")
print("Feature Names:", cv.get_feature_names_out())
print("TF Matrix:\n", tf_matrix.toarray())

# ----------------------------------------------------------
# STEP 9: TF-IDF Representation
# ----------------------------------------------------------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

print("\nTF-IDF Representation:")
print("Feature Names:", tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())