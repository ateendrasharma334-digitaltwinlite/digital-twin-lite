import sqlite3

def get_connection():
    conn = sqlite3.connect(
        "sensor_data.db",
        check_same_thread=False,
        timeout=30
    )
    return conn


def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        temperature REAL,
        vibration REAL,
        pressure REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maintenance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        machine TEXT,
        issue TEXT,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()