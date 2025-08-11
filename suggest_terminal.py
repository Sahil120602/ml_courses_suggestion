import mysql.connector
import pandas as pd
import sys
import msvcrt

# ===== Database Connection =====
db_config = {
    'host': 'localhost',
    'user': 'root',           # Update with your MySQL username
    'password': '',           # Update with your MySQL password
    'database': 'course_db'   # Update with your database name
}

def load_courses_from_db():
    try:
        conn = mysql.connector.connect(**db_config)
        query = "SELECT course_name FROM courses WHERE status = 'Active'"
        courses_df = pd.read_sql(query, conn)
        conn.close()
        
        # Add preprocessed column
        courses_df['course_preproc'] = courses_df['course_name'].str.lower().str.strip()
        return courses_df
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        print("Falling back to CSV file...")
        # Fallback to CSV
        courses_df = pd.read_csv("ml_suggestion_module/courses_lookup.csv")
        return courses_df

courses_df = load_courses_from_db()

def suggest_from_db(query, k=None):
    q_proc = query.lower().strip()
    results = []
    
    for idx, row in courses_df.iterrows():
        course_name = row["course_name"]
        cname_norm = row["course_preproc"]
        
        # Simple prefix and word matching
        score = 0
        # Remove dots and special chars for better matching
        clean_name = cname_norm.replace('.', '').replace('(', ' ').replace(')', ' ')
        clean_words = clean_name.split()
        
        if cname_norm.startswith(q_proc) or clean_name.startswith(q_proc):
            score = 0.9  # High score for prefix match
        elif any(word.startswith(q_proc) for word in clean_words):
            score = 0.7  # Good score for word start match
        elif q_proc in cname_norm or q_proc in clean_name:
            score = 0.3  # Lower score for substring match
        
        if score > 0:
            results.append((course_name, score))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    if k:
        results = results[:k]
    return results



import sys
import msvcrt

def get_realtime_suggestions(query):
    if len(query) == 0:
        return []
    suggestions = suggest_from_db(query, k=20)
    return suggestions

# ===== Real-time Interactive Mode =====
print("🎓 Real-time Degree Suggestion")
print("Start typing (ESC to exit)\n")

current_input = ""
while True:
    print(f"\rQuery: {current_input}", end="", flush=True)
    
    if msvcrt.kbhit():
        key = msvcrt.getch()
        
        if key == b'\x1b':  # ESC key
            break
        elif key == b'\r':  # Enter key
            print("\n")
            continue
        elif key == b'\x08':  # Backspace
            if current_input:
                current_input = current_input[:-1]
        elif 32 <= ord(key) <= 126:  # Printable characters
            current_input += key.decode('utf-8')
        
        # Clear screen and show suggestions
        print("\033[2J\033[H", end="")  # Clear screen
        print("🎓 Real-time Degree Suggestion")
        print("Start typing (ESC to exit)\n")
        print(f"Query: {current_input}")
        
        if current_input:
            suggestions = get_realtime_suggestions(current_input)
            if suggestions:
                print(f"\nSuggestions ({len(suggestions)}):")
                for name, score in suggestions:  # Show all suggestions
                    print(f"  {name} ({int(score*100)}%)")
            else:
                print("\nNo matches found")
        print("\n" + "-"*50)

print("\nGoodbye!")

 
