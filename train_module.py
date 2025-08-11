import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ===== Config =====
DATA_PATH = "Courses.csv"   # Your CSV file
OUT_DIR = Path("ml_suggestion_module")
OUT_DIR.mkdir(exist_ok=True)

# ===== Load Data =====
df = pd.read_csv(DATA_PATH)
if "course_name" not in df.columns:
    raise ValueError("CSV must contain 'course_name' column.")

def preprocess(text: str) -> str:
    """Lowercase, strip, normalize spaces."""
    return " ".join(str(text).lower().strip().split())

df["course_preproc"] = df["course_name"].apply(preprocess)

# ===== Train TF-IDF Vectorizer =====
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
X = vectorizer.fit_transform(df["course_preproc"])

# ===== Train Nearest Neighbors =====
nn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
nn.fit(X)

# ===== Save Artifacts =====
pickle.dump(vectorizer, open(OUT_DIR / "vectorizer.pkl", "wb"))
pickle.dump(nn, open(OUT_DIR / "nn_model.pkl", "wb"))
df[["course_name", "course_preproc"]].to_csv(OUT_DIR / "courses_lookup.csv", index=False)

print(f"✅ Model trained and saved in: {OUT_DIR}")
