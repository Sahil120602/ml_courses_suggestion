# suggest_db_realtime.py
import os
import pickle
import re
import pymysql
import pandas as pd
import numpy as np
import msvcrt
from pathlib import Path

# ===== CONFIG - EDIT THESE =====
BASE_DIR = r"C:\Users\Dell\Desktop\course_suggestion"   # folder where ml_suggestion_module exists
VECTOR_PATH = os.path.join(BASE_DIR, "ml_suggestion_module", "vectorizer.pkl")
NN_PATH = os.path.join(BASE_DIR, "ml_suggestion_module", "nn_model.pkl")

DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = "root"                # <- put your MySQL password here
DB_NAME = "resume_analyzer"
DB_TABLE = "ups_courses"
DB_STATUS_COL = "status"        # status column to filter active rows
DB_STATUS_VAL = "Active"        # value representing active courses
DB_COURSE_COL = "course_name"   # column name for the course title
# ================================

# ===== Helpers =====
def preprocess(text: str) -> str:
    """Normalize similar to training: lowercase, remove parenthesis content, punctuation -> spaces, collapse spaces."""
    t = str(text).lower()
    t = re.sub(r"\([^)]*\)", "", t)           # remove parentheses content
    t = re.sub(r"[\./,]", " ", t)             # turn punctuation into spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

def dynamic_min_score(q_norm: str) -> float:
    """Lower threshold for short queries, higher for long queries."""
    L = len(q_norm)
    if L <= 1:
        return 0.0
    if L == 2:
        return 0.06
    if L == 3:
        return 0.12
    return 0.18

# ===== Load ML artifacts =====
if not os.path.exists(VECTOR_PATH) or not os.path.exists(NN_PATH):
    raise SystemExit(f"Missing model files. Place vectorizer.pkl and nn_model.pkl under {os.path.join(BASE_DIR, 'ml_suggestion_module')}")

with open(VECTOR_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(NN_PATH, "rb") as f:
    nn_model = pickle.load(f)

# ===== Fetch courses from DB =====
print("Connecting to MySQL and loading course list...")
conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME, charset='utf8mb4')
try:
    sql = f"SELECT `{DB_COURSE_COL}` FROM `{DB_TABLE}` WHERE `{DB_STATUS_COL}` = %s"
    df = pd.read_sql(sql, conn, params=(DB_STATUS_VAL,))
finally:
    conn.close()

if df.empty:
    raise SystemExit("No courses loaded from DB. Check your table/credentials/status filter.")

# keep original and preprocessed columns
df[DB_COURSE_COL] = df[DB_COURSE_COL].astype(str)
df["course_preproc"] = df[DB_COURSE_COL].apply(preprocess)

# Vectorize all course texts ONCE (use same vectorizer used during training)
course_texts = df["course_preproc"].tolist()
course_vectors = vectorizer.transform(course_texts)  # sparse matrix

# Precompute token lists and no-dots versions for boosting checks
df["course_no_dots"] = df["course_preproc"].str.replace(".", "")
df["tokens"] = df["course_preproc"].str.split()

print(f"Loaded {len(df)} active courses from DB. Ready for realtime suggestions.\n")

# ===== Suggest function using NN model =====
from sklearn.metrics.pairwise import cosine_similarity

def suggest_ml(query: str, k_display=20):
    q_norm = preprocess(query)
    if q_norm == "":
        return []

    # vectorize query
    q_vec = vectorizer.transform([q_norm])

    # get neighbors (ask for all so we can filter by dynamic threshold)
    n_candidates = max( min(len(df), 2000), 1 )
    distances, indices = nn_model.kneighbors(q_vec, n_neighbors=n_candidates, return_distance=True)

    distances = distances[0]
    indices = indices[0]

    min_score = dynamic_min_score(q_norm)
    results = []

    for dist, idx in zip(distances, indices):
        sim = float(1 - dist)  # similarity in [0,1]
        cname = df.iloc[idx][DB_COURSE_COL]
        cname_norm = df.iloc[idx]["course_preproc"]
        cname_no_dots = df.iloc[idx]["course_no_dots"]
        tokens = df.iloc[idx]["tokens"]

        # Boosts (to prefer prefix / token-start matches)
        boost = 0.0
        if cname_norm.startswith(q_norm) or (q_norm.replace(".", "") and cname_no_dots.startswith(q_norm.replace(".", ""))):
            boost += 0.28
        if any(tok.startswith(q_norm) or tok.startswith(q_norm.replace(".", "")) for tok in tokens):
            boost += 0.12

        final_score = min(sim + boost, 0.9999)

        if final_score >= min_score:
            results.append((cname, final_score))

    # sort by score desc then name
    results.sort(key=lambda x: (-x[1], x[0]))

    # Return top k_display results (or all if k_display is None)
    if k_display is not None:
        return results[:k_display]
    return results

# ===== Real-time interactive loop (msvcrt) =====
print("🎓 Real-time Degree Suggestion (ML + MySQL)")
print("Start typing (ESC to exit)\n")

current_input = ""
while True:
    print(f"\rQuery: {current_input}", end="", flush=True)

    if msvcrt.kbhit():
        key = msvcrt.getch()

        if key == b'\x1b':  # ESC key
            break
        elif key == b'\r':  # Enter
            print("\n")
            continue
        elif key == b'\x08':  # Backspace
            if current_input:
                current_input = current_input[:-1]
        elif 32 <= ord(key) <= 126:  # printable
            current_input += key.decode('utf-8')

        # Clear screen
        print("\033[2J\033[H", end="")
        print("🎓 Real-time Degree Suggestion (ML + MySQL)")
        print("Start typing (ESC to exit)\n")
        print(f"Query: {current_input}")

        if current_input:
            try:
                suggestions = suggest_ml(current_input, k_display=12)  # show top 12
            except Exception as e:
                print(f"\nError while suggesting: {e}")
                suggestions = []

            if suggestions:
                print(f"\nSuggestions ({len(suggestions)}):")
                for name, score in suggestions:
                    print(f"  {name}  ({int(round(score*100))}%)")
            else:
                print("\nNo matches found")
        print("\n" + "-"*50)

print("\nGoodbye!")
