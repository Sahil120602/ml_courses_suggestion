from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import sys
import os

# Add current directory to path so we can import the ML model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your ML model functions
from course_suggestion_realtime import fetch_courses_from_db, get_suggestions

app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend integration

# Global variable to store courses data
courses_df = None

def load_courses():
    """Load courses from database once at startup"""
    global courses_df
    if courses_df is None:
        print("Loading courses from database...")
        courses_df = fetch_courses_from_db()
        print(f"Loaded {len(courses_df)} courses for API")
    return courses_df

# ===== API Endpoints =====

@app.route('/api/suggest', methods=['GET'])
def suggest():
    """
    Course suggestion API - exactly like terminal
    Just pass any query and get suggestions
    """
    start_time = time.time()
    
    # Get user's search query
    query = request.args.get('query', '').strip()
    limit = request.args.get('limit', 8, type=int)
    
    # Empty query = no suggestions (like terminal)
    if not query:
        return jsonify({
            'query': '',
            'suggestions': [],
            'count': 0
        })
    
    # Get suggestions using ML model
    current_courses = load_courses()
    suggestions = get_suggestions(query, current_courses, top_k=limit)
    
    # Response time
    response_time = round((time.time() - start_time) * 1000, 2)
    
    # Simple response - just like terminal
    return jsonify({
        'query': query,
        'suggestions': [
            {
                'course_name': name,
                'confidence_percentage': score
            }
            for name, score in suggestions
        ],
        'count': len(suggestions),
        'response_time_ms': response_time
    })

@app.route('/', methods=['GET'])
def home():
    """Simple API info"""
    return jsonify({
        'title': '🎓 Course Suggestion API',
        'description': 'Type any search term and get course suggestions',
        'usage': 'GET /api/suggest?query=YOUR_SEARCH_HERE',
        'note': 'Replace YOUR_SEARCH_HERE with anything you want to search for'
    })

if __name__ == '__main__':
    print("🎓 Starting Course Suggestion API...")
    print("📡 API: http://localhost:5000/api/suggest?query=YOUR_SEARCH")
    print("💡 In Postman: Just change YOUR_SEARCH to anything you want!")
    print("   Example: http://localhost:5000/api/suggest?query=engineering")
    
    # Load courses at startup
    load_courses()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
