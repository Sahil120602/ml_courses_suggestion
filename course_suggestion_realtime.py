# course_suggestion_realtime.py
import warnings
import pickle
import pandas as pd
import msvcrt
import threading
import time
import hashlib
import subprocess
from pathlib import Path
from sqlalchemy import create_engine

# ========== CONFIG ==========
DB_CONFIG = {
    "user": "root",
    "password": "root",
    "host": "localhost",
    "port": 3306,
    "database": "resume_analyzer"
}

TABLE_NAME = "ups_courses"
COURSE_COL = "course_name"
POLL_SECONDS = 5

BASE = Path("ml_suggestion_module")
CHAR_VEC_FILE = BASE / "char_vectorizer.pkl"
WORD_VEC_FILE = BASE / "word_vectorizer.pkl"
MODEL_FILE = BASE / "nn_model.pkl"

char_vectorizer = pickle.load(open(CHAR_VEC_FILE, "rb"))
word_vectorizer = pickle.load(open(WORD_VEC_FILE, "rb"))
nn = pickle.load(open(MODEL_FILE, "rb"))

def get_courses_hash():
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    df = pd.read_sql(f"SELECT `{COURSE_COL}` FROM `{TABLE_NAME}` WHERE status = 'Active'", engine)
    courses_str = ",".join(sorted(df[COURSE_COL].astype(str).tolist()))
    return hashlib.md5(courses_str.encode()).hexdigest()

def retrain_and_reload():
    subprocess.run(["python", "retrain_course_model.py"])
    global char_vectorizer, word_vectorizer, nn
    char_vectorizer = pickle.load(open(CHAR_VEC_FILE, "rb"))
    word_vectorizer = pickle.load(open(WORD_VEC_FILE, "rb"))
    nn = pickle.load(open(MODEL_FILE, "rb"))
    print("✅ Model reloaded after retraining.")

def monitor_courses(interval=300):
    last_hash = get_courses_hash()
    while True:
        time.sleep(interval)
        current_hash = get_courses_hash()
        if current_hash != last_hash:
            print("🔄 Course table changed. Retraining model...")
            retrain_and_reload()
            last_hash = current_hash

threading.Thread(target=monitor_courses, daemon=True).start()

def fetch_courses_from_db():
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    query = f"SELECT `id`, `{COURSE_COL}` FROM `{TABLE_NAME}` WHERE status = 'Active'"
    df = pd.read_sql(query, engine)
    df = df[df[COURSE_COL].notna()].drop_duplicates(subset=[COURSE_COL]).reset_index(drop=True)
    df[COURSE_COL] = df[COURSE_COL].astype(str).str.strip()
    print(f"Loaded {len(df)} active courses from database")
    return df

def get_suggestions(query, courses_df, top_k=8):
    if not query.strip():
        return []

    max_neighbors = min(top_k * 2, len(courses_df))
    if max_neighbors == 0:
        return []

    try:
        def normalize_text(text):
            import re
            text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower().strip())
            return " ".join(text.split())

        query_normalized = normalize_text(query)
        query_no_dots = query_normalized.replace('.', '')

        from scipy.sparse import hstack
        X_query_char = char_vectorizer.transform([query_normalized])
        X_query_word = word_vectorizer.transform([query_normalized])
        X_query = hstack([X_query_char, X_query_word])

        distances, indices = nn.kneighbors(X_query, n_neighbors=max_neighbors)
        suggestions = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(courses_df):
                course_name = courses_df.iloc[idx][COURSE_COL]
                course_id = int(courses_df.iloc[idx]['id'])
                ml_score = max(0, int((1 - dist) * 100))

                query_lower = query.lower().strip()
                query_clean = query_no_dots
                course_lower = course_name.lower().strip()
                course_clean = normalize_text(course_name)
                course_no_dots = course_clean.replace('.', '')

                boost = 0
                if query_clean == course_clean:
                    boost = 100  # Exact match
                elif course_lower.startswith(query_lower) or course_no_dots.startswith(query_clean):
                    boost = 90  # Prefix match
                elif any(word.startswith(query_clean) or word.startswith(query_no_dots) for word in course_clean.split()):
                    boost = 70  # Word start match
                elif query_clean in course_clean or query_no_dots in course_no_dots:
                    boost = 30  # Substring match

                final_score = min(100, ml_score + boost)

                is_relevant = False
                # For queries >= 3 chars (like "diploma"), require a text-based match
                if len(query_clean) >= 3:
                    if boost > 0:  # Must have prefix, word-start, or substring match
                        is_relevant = True
                else:
                    # For shorter queries, allow high ML scores or text matches
                    if boost > 0 or ml_score >= 70:
                        is_relevant = True

                # Additional check for longer queries to ensure relevance
                if is_relevant and len(query_clean) >= 3:
                    has_connection = (
                        query_clean in course_clean or
                        query_no_dots in course_no_dots or
                        any(word.startswith(query_clean) or word.startswith(query_no_dots) for word in course_clean.split()) or
                        query_clean == course_clean  # Exact match
                    )
                    # Special case for short abbreviations (e.g., "bttm")
                    if not has_connection and len(query_clean) <= 5:
                        if ml_score >= 80:  # Higher threshold for ML-only matches
                            has_connection = True
                    if not has_connection:
                        is_relevant = False

                # Only include if relevant and score is above threshold
                if is_relevant and final_score >= 40:
                    suggestions.append((course_name, course_id, final_score))
            else:
                continue

        suggestions.sort(key=lambda x: (-x[2], x[0]))
        return suggestions[:top_k]
        
    except Exception as e:
        print(f"Error in suggestions: {e}")
        return []

def realtime_mode():
    courses_df = fetch_courses_from_db()
    print("🎓 Real-time Degree Suggestion (ML + MySQL Auto-Update)")
    print("Start typing (ESC to exit)\n")
    current_input = ""

    while True:
        print(f"\rQuery: {current_input}", end="", flush=True)

        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                break
            elif key == b'\r':
                print("\n")
                continue
            elif key == b'\x08':
                if current_input:
                    current_input = current_input[:-1]
            elif 32 <= ord(key) <= 126:
                current_input += key.decode('utf-8')

            print("\033[2J\033[H", end="")
            print("🎓 Real-time Degree Suggestion (ML + MySQL Auto-Update)")
            print("Start typing (ESC to exit)\n")
            print(f"Query: {current_input}")

            if current_input:
                suggestions = get_suggestions(current_input, courses_df)
                if suggestions:
                    print(f"\nSuggestions ({len(suggestions)}):")
                    for name, course_id, score in suggestions:
                        print(f"  {name} ({score}%) [ID: {course_id}]")
                else:
                    print("\nNo matches found")
            print("\n" + "-" * 50)

if __name__ == "__main__":
    realtime_mode()