from rapidfuzz import process, fuzz
# ml_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import sys
import os
import threading
from course_suggestion_realtime import monitor_courses, fetch_courses_from_db, get_suggestions

# Start background thread to monitor DB changes and reload model/courses every 5 minutes
threading.Thread(target=monitor_courses, args=(300,), daemon=True).start()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
CORS(app)

courses_df = None

def load_courses():
    """Load courses from database once at startup"""
    global courses_df
    if courses_df is None:
        print("Loading courses from database...")
        courses_df = fetch_courses_from_db()
        print(f"Loaded {len(courses_df)} courses for API")
    return courses_df

@app.route('/api/suggest', methods=['GET'])
def suggest():
    start_time = time.time()
    query = request.args.get('query', '').strip()
    

    # Query validation: only return error if query has no alphanumeric characters at all
    import re
    if not query or not re.search(r'[a-zA-Z0-9]', query):
        return jsonify({
            'query': query,
            'suggestions': [],
            'count': 0,
            'message': 'Please enter a valid query containing at least one letter or number.'
        })

    # Adjusted dynamic limit: more suggestions for "diploma"
    qlen = len(query)
    if qlen <= 1:
        limit = 30
    elif qlen <= 3:
        limit = 20
    elif qlen <= 6:
        limit = 10
    else:
        limit = 8  # Increased from 3 to 8 for queries like "diploma"

    current_courses = load_courses()
    ml_suggestions = get_suggestions(query, current_courses, top_k=limit)

    # Improved rule-based fallback (aligned with suggest_terminal.py)
    def normalize_text(text):
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

    query_norm = normalize_text(query)
    query_no_dots = query_norm.replace('.', '')
    fallback = []

    for idx, row in current_courses.iterrows():
        course_name = row["course_name"]
        course_id = int(row["id"])
        course_norm = normalize_text(course_name)
        course_no_dots = course_norm.replace('.', '')
        
        score = 0
        if course_norm.startswith(query_norm) or course_no_dots.startswith(query_no_dots):
            score = 90  # High score for prefix match
        elif any(word.startswith(query_norm) or word.startswith(query_no_dots) for word in course_norm.split()) or \
             any(word.startswith(query_norm) or word.startswith(query_no_dots) for word in course_no_dots.split()):
            score = 70  # Good score for word start match
        elif query_norm in course_norm or query_no_dots in course_no_dots:
            score = 30  # Lower score for substring match

        if score > 0:
            fallback.append((course_name, course_id, score))

    # Merge ML and fallback, prefer higher confidence
    all_suggestions = {(name, cid): score for name, cid, score in fallback}
    for name, cid, score in ml_suggestions:
        if (name, cid) not in all_suggestions or score > all_suggestions[(name, cid)]:
            all_suggestions[(name, cid)] = score


    suggestions = [(name, cid, score) for (name, cid), score in all_suggestions.items()]
    suggestions.sort(key=lambda x: (-x[2], x[0]))
    suggestions = suggestions[:limit]

    # Fuzzy fallback if no suggestions (abbreviation/typo handling)
    if not suggestions:
        course_names = current_courses["course_name"].tolist()
        matches = process.extract(
            query,
            course_names,
            scorer=fuzz.WRatio,
            limit=limit
        )
        fuzzy_results = []
        for name, score, idx in matches:
            if score < 60:
                continue
            course_id = int(current_courses.iloc[idx]["id"])
            fuzzy_results.append((name, course_id, score))
        suggestions = fuzzy_results

    response_time = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'query': query,
        'suggestions': [
            {
                'course_id': course_id,
                'course_name': name,
                'confidence_percentage': score
            }
            for name, course_id, score in suggestions
        ],
        'count': len(suggestions),
        'response_time_ms': response_time
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'title': '🎓 Course Suggestion API',
        'description': 'Type any search term and get course suggestions',
        'usage': 'GET /api/suggest?query=YOUR_SEARCH_HERE',
        'note': 'Replace YOUR_SEARCH_HERE with anything you want to search for'
    })

if __name__ == '__main__':
    print("🎓 Starting Course Suggestion API...")
    print("📡 API: http://localhost:5002/api/suggest?query=YOUR_SEARCH")
    print("💡 In Postman: Just change YOUR_SEARCH to anything you want!")
    print("   Example: http://localhost:5002/api/suggest?query=engineering")
    load_courses()
    app.run(debug=True, host='0.0.0.0', port=5002)