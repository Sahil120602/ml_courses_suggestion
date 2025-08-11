import warnings
import pickle
import pandas as pd
import msvcrt
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
POLL_SECONDS = 5  # Time interval to re-fetch DB

# ML model files
BASE = Path("ml_suggestion_module")
VEC_FILE = BASE / "vectorizer.pkl"
MODEL_FILE = BASE / "nn_model.pkl"

# ========== LOAD ML MODEL ==========
vectorizer = pickle.load(open(VEC_FILE, "rb"))
nn = pickle.load(open(MODEL_FILE, "rb"))

# ========== FETCH COURSES ==========
def fetch_courses_from_db():
    """Fetch fresh course list from MySQL using SQLAlchemy."""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    # Only get active courses to match training data
    query = f"SELECT `{COURSE_COL}` FROM `{TABLE_NAME}` WHERE status = 'Active'"
    df = pd.read_sql(query, engine)
    df = df[df[COURSE_COL].notna()].drop_duplicates(subset=[COURSE_COL]).reset_index(drop=True)
    df[COURSE_COL] = df[COURSE_COL].astype(str).str.strip()
    print(f"Loaded {len(df)} active courses from database")
    return df

# ========== SUGGESTION FUNCTION ==========
def get_suggestions(query, courses_df, top_k=8):
    if not query.strip():
        return []
    
    # Safety check: ensure we don't request more neighbors than available courses
    max_neighbors = min(top_k, len(courses_df))
    if max_neighbors == 0:
        return []
    
    try:
        X_query = vectorizer.transform([query])
        distances, indices = nn.kneighbors(X_query, n_neighbors=max_neighbors)
        suggestions = []
        
        for dist, idx in zip(distances[0], indices[0]):
            # Safety check: ensure the index is within DataFrame bounds
            if idx < len(courses_df):
                course_name = courses_df.iloc[idx][COURSE_COL]
                ml_score = max(0, int((1 - dist) * 100))
                
                # Add prefix matching boost for better relevance
                query_lower = query.lower().strip()
                course_lower = course_name.lower().strip()
                
                # Boost score if course starts with query
                if course_lower.startswith(query_lower):
                    final_score = min(100, ml_score + 30)  # +30 boost for prefix match
                # Boost if any word in course starts with query
                elif any(word.startswith(query_lower) for word in course_lower.split()):
                    final_score = min(100, ml_score + 20)  # +20 boost for word start match
                # Slight boost if query is at start after removing dots/spaces
                elif course_lower.replace('.', '').replace(' ', '').startswith(query_lower):
                    final_score = min(100, ml_score + 15)  # +15 boost for cleaned prefix match
                else:
                    final_score = ml_score
                
                suggestions.append((course_name, final_score))
            else:
                # Skip invalid indices
                continue
        
        # Sort by final score (highest first), then alphabetically
        suggestions.sort(key=lambda x: (-x[1], x[0]))
        return suggestions
        
    except Exception as e:
        print(f"Error in suggestions: {e}")
        return []

# ========== REALTIME MODE ==========
def realtime_mode():
    courses_df = fetch_courses_from_db()
    print("🎓 Real-time Degree Suggestion (ML + MySQL Auto-Update)")
    print("Start typing (ESC to exit)\n")
    current_input = ""

    while True:
        print(f"\rQuery: {current_input}", end="", flush=True)

        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                break
            elif key == b'\r':  # Enter
                print("\n")
                continue
            elif key == b'\x08':  # Backspace
                if current_input:
                    current_input = current_input[:-1]
            elif 32 <= ord(key) <= 126:  # Printable
                current_input += key.decode('utf-8')

            # Clear terminal
            print("\033[2J\033[H", end="")
            print("🎓 Real-time Degree Suggestion (ML + MySQL Auto-Update)")
            print("Start typing (ESC to exit)\n")
            print(f"Query: {current_input}")

            if current_input:
                suggestions = get_suggestions(current_input, courses_df)
                if suggestions:
                    print(f"\nSuggestions ({len(suggestions)}):")
                    for name, score in suggestions:
                        print(f"  {name} ({score}%)")
                else:
                    print("\nNo matches found")
            print("\n" + "-" * 50)

if __name__ == "__main__":
    realtime_mode()
