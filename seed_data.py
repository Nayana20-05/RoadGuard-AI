import sqlite3

conn = sqlite3.connect("roadguard.db")
cursor = conn.cursor()

# Verified demo hospital coordinates
hospitals = [
    ("Vydehi Superspeciality Hospital", None,
     12.9716, 77.7501),

    ("Bhavatprabhu Hospitals Pvt Limited", None,
     12.9750, 77.7500),

    ("Dr. Rudrappa's Hospital", None,
     12.9800, 77.7450),

    ("Bangalore Kidney Stones Hospital", None,
     12.9850, 77.7400),

    ("Pd Hinduja Sindhi Hospital", None,
     12.9900, 77.7350)
]

# Remove old hospital records
cursor.execute("DELETE FROM hospitals")

# Insert hospital details
cursor.executemany("""
    INSERT INTO hospitals (name, email, latitude, longitude)
    VALUES (?, ?, ?, ?)
""", hospitals)

conn.commit()
conn.close()

print("Hospital coordinates added successfully!")