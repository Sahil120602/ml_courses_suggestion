import mysql.connector

# Database config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'resume_analyzer'
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    # Test 1: Count total courses
    cursor.execute("SELECT COUNT(*) FROM ups_courses WHERE status = 'Active'")
    count = cursor.fetchone()[0]
    print(f"Total active courses: {count}")
    
    # Test 2: Find architecture courses
    cursor.execute("SELECT course_name FROM ups_courses WHERE status = 'Active' AND LOWER(course_name) LIKE '%arch%'")
    arch_courses = cursor.fetchall()
    print(f"Architecture courses: {[course[0] for course in arch_courses]}")
    
    # Test 3: Find courses starting with 'b'
    cursor.execute("SELECT course_name FROM ups_courses WHERE status = 'Active' AND LOWER(course_name) LIKE 'b%' LIMIT 10")
    b_courses = cursor.fetchall()
    print(f"Courses starting with 'b': {[course[0] for course in b_courses]}")
    
    # Test 4: Check status values
    cursor.execute("SELECT DISTINCT status FROM ups_courses")
    statuses = cursor.fetchall()
    print(f"Status values in table: {[status[0] for status in statuses]}")
    
    cursor.close()
    conn.close()
    
except mysql.connector.Error as e:
    print(f"Database error: {e}")