import sqlite3

DB_NAME = "roadguard.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Accident records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            severity TEXT,
            accident_time TEXT,
            ambulance_id TEXT,
            hospital_name TEXT,
            ambulance_alert_sent INTEGER DEFAULT 0,
            hospital_alert_sent INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Pending'
        )
    """)

    # Ambulance records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ambulances (
            ambulance_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            latitude REAL,
            longitude REAL,
            status TEXT DEFAULT 'Available'
        )
    """)

    # Hospital records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hospitals (
            hospital_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            latitude REAL,
            longitude REAL
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
    print("RoadGuard AI database initialized successfully!")