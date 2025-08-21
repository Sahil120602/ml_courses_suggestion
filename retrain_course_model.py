import pandas as pd
import mysql.connector
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors



# ==== Paths ====
BASE_DIR = Path(r"C:\Users\Dell\Desktop\course_suggestion\ml_suggestion_module")
VECTOR_FILE = BASE_DIR / "vectorizer.pkl"
MODEL_FILE = BASE_DIR / "nn_model.pkl"
LOOKUP_FILE = BASE_DIR / "courses_lookup.csv"

# ==== DB config ====
DB_CONFIG = {
    "host": "localhost",
    "user": "root",           # change to your MySQL username
    "password": "root",   # change to your MySQL password
    "database": "resume_analyzer"
}
TABLE_NAME = "ups_courses"
COURSE_COL = "course_name"

# ==== Fetch courses from DB ====
conn = mysql.connector.connect(**DB_CONFIG)
query = f"SELECT {COURSE_COL} FROM {TABLE_NAME}"
df = pd.read_sql(query, conn)
conn.close()

# Normalize course names for training (same as suggestion logic)
def normalize_text(text):
    """Expand abbreviations, remove special chars, normalize spaces, lowercase"""
    import re
    ABBREV = {
        'mgmt': 'management',
        'engg': 'engineering',
        'tech': 'technology',
        'comm': 'commerce',
        'sci': 'science',
        'phy': 'physics',
        'chem': 'chemistry',
        'bio': 'biology',
        # add more as needed
    }
    text = str(text).lower().strip()
    for abbr, full in ABBREV.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return " ".join(text.split())

df[COURSE_COL] = df[COURSE_COL].astype(str).str.strip()
df["course_normalized"] = df[COURSE_COL].apply(normalize_text)
courses = df["course_normalized"].tolist()  # Use normalized text for training


# ==== Train both char n-gram and word-level TF-IDF vectorizers ====
from sklearn.pipeline import FeatureUnion

char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))

X_char = char_vectorizer.fit_transform(courses)
X_word = word_vectorizer.fit_transform(courses)

# Combine both feature spaces (horizontally)
from scipy.sparse import hstack
X_combined = hstack([X_char, X_word])

nn_model = NearestNeighbors(n_neighbors=len(courses), metric="cosine")
nn_model.fit(X_combined)

# Save both vectorizers
with open(BASE_DIR / "char_vectorizer.pkl", "wb") as f:
    pickle.dump(char_vectorizer, f)
with open(BASE_DIR / "word_vectorizer.pkl", "wb") as f:
    pickle.dump(word_vectorizer, f)

# ==== Save artifacts ====

BASE_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(nn_model, f)

df.to_csv(LOOKUP_FILE, index=False)

print(f"✅ Model trained on {len(courses)} courses from DB and saved to {BASE_DIR}")
