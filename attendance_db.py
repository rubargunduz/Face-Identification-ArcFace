import sqlite3
from datetime import datetime

DB_PATH = "attendance_log.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            datetime TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO attendance (name, datetime) VALUES (?, ?)", (name, now))
    conn.commit()
    conn.close()
    print(f"[LOGGED] {name} at {now}")

# Initialize DB on import
init_db()
