import sqlite3
from geopy.distance import geodesic

DB_NAME = "roadguard.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def find_nearest_ambulance(accident_lat, accident_lon):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ambulance_id, name, email, latitude, longitude, status
        FROM ambulances
        WHERE status = 'Available'
    """)

    ambulances = cursor.fetchall()
    conn.close()

    if not ambulances:
        return None

    nearest = None
    min_distance = float("inf")

    for ambulance in ambulances:
        ambulance_id, name, email, lat, lon, status = ambulance

        distance = geodesic(
            (accident_lat, accident_lon),
            (lat, lon)
        ).km

        if distance < min_distance:
            min_distance = distance
            nearest = {
                "ambulance_id": ambulance_id,
                "name": name,
                "email": email,
                "latitude": lat,
                "longitude": lon,
                "distance_km": round(distance, 2)
            }

    return nearest


def find_nearest_hospital(accident_lat, accident_lon):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, email, latitude, longitude
        FROM hospitals
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
    """)

    hospitals = cursor.fetchall()
    conn.close()

    if not hospitals:
        return None

    nearest = None
    min_distance = float("inf")

    for hospital in hospitals:
        name, email, lat, lon = hospital

        distance = geodesic(
            (accident_lat, accident_lon),
            (lat, lon)
        ).km

        if distance < min_distance:
            min_distance = distance
            nearest = {
                "name": name,
                "email": email,
                "latitude": lat,
                "longitude": lon,
                "distance_km": round(distance, 2)
            }

    return nearest


def google_maps_link(latitude, longitude):
    return (
        f"https://www.google.com/maps/search/?api=1"
        f"&query={latitude},{longitude}"
    )


if __name__ == "__main__":
    # Demo accident location
    accident_lat = 12.9716
    accident_lon = 77.5946

    ambulance = find_nearest_ambulance(
        accident_lat, accident_lon
    )

    hospital = find_nearest_hospital(
        accident_lat, accident_lon
    )

    print("\nNearest available ambulance:")
    print(ambulance)

    print("\nNearest hospital:")
    print(hospital)

    print("\nAccident location:")
    print(google_maps_link(accident_lat, accident_lon))

    if hospital:
        print("\nHospital map:")
        print(google_maps_link(
            hospital["latitude"],
            hospital["longitude"]
        ))