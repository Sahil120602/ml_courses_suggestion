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

df[COURSE_COL] = df[COURSE_COL].astype(str).str.strip()
courses = df[COURSE_COL].tolist()

# ==== Train model ====
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = vectorizer.fit_transform(courses)

nn_model = NearestNeighbors(n_neighbors=len(courses), metric="cosine")
nn_model.fit(X)

# ==== Save artifacts ====
BASE_DIR.mkdir(parents=True, exist_ok=True)
with open(VECTOR_FILE, "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(nn_model, f)

df.to_csv(LOOKUP_FILE, index=False)

print(f"✅ Model trained on {len(courses)} courses from DB and saved to {BASE_DIR}")
