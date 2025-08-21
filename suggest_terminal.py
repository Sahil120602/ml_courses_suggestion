import pickle
import pandas as pd
from pathlib import Path
import sys
import msvcrt

# ===== Load Artifacts =====
BASE = Path("ml_suggestion_module")
vectorizer = pickle.load(open(BASE / "vectorizer.pkl", "rb"))
nn = pickle.load(open(BASE / "nn_model.pkl", "rb"))
courses_df = pd.read_csv(BASE / "courses_lookup.csv")

def suggest(query, k=None):
    q_proc = query.lower().strip()
    # Remove dots for better matching (e.g., "b.e" -> "be")
    q_proc_no_dots = q_proc.replace('.', '')
    results = []
    
    for idx, row in courses_df.iterrows():
        course_name = row["course_name"]
        cname_norm = row["course_normalized"]
        # Also create a version without dots for comparison
        cname_no_dots = cname_norm.replace('.', '')
        
        # Enhanced matching logic
        score = 0
        
        # Check exact prefix matches (with and without dots)
        if cname_norm.startswith(q_proc) or cname_no_dots.startswith(q_proc_no_dots):
            score = 0.9  # High score for prefix match
        # Check if any word starts with query (with and without dots)
        elif any(word.startswith(q_proc) or word.startswith(q_proc_no_dots) for word in cname_norm.split()) or \
             any(word.startswith(q_proc) or word.startswith(q_proc_no_dots) for word in cname_no_dots.split()):
            score = 0.7  # Good score for word start match
        # Check substring matches
        elif q_proc in cname_norm or q_proc_no_dots in cname_no_dots:
            score = 0.3  # Lower score for substring match
        
        if score > 0:
            results.append((course_name, score))
    
    # Sort by score, then alphabetically
    results.sort(key=lambda x: (-x[1], x[0]))
    if k:
        results = results[:k]
    return results



import sys
import msvcrt

def get_realtime_suggestions(query):
    if len(query) == 0:
        return []
    suggestions = suggest(query, k=20)
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
                for name, score in suggestions[:8]:  # Show top 8
                    print(f"  {name} ({int(score*100)}%)")
            else:
                print("\nNo matches found")
        print("\n" + "-"*50)

print("\nGoodbye!")

 
