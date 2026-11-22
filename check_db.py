import sqlite3

conn = sqlite3.connect("roadguard.db")
cursor = conn.cursor()

for table in ["ambulances", "hospitals", "accidents"]:
    print(f"\n--- {table.upper()} ---")
    cursor.execute(f"SELECT * FROM {table}")
    
    for row in cursor.fetchall():
        print(row)

conn.close()