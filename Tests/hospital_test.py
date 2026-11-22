import requests
from geopy.distance import geodesic

accident_lat = 12.9716
accident_lon = 77.5946

query = f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:5000,{accident_lat},{accident_lon});
  way["amenity"="hospital"](around:5000,{accident_lat},{accident_lon});
  relation["amenity"="hospital"](around:5000,{accident_lat},{accident_lon});
);
out center;
"""

url = "https://overpass-api.de/api/interpreter"

headers = {
    "User-Agent": "RoadGuardAI/1.0"
}

response = requests.post(
    url,
    data={"data": query},
    headers=headers,
    timeout=40
)

print("Status:", response.status_code)

data = response.json()

hospitals = []

for place in data["elements"]:

    tags = place.get("tags", {})
    name = tags.get("name", "Unnamed Hospital")

    if place["type"] == "node":
        lat = place["lat"]
        lon = place["lon"]

    else:
        center = place.get("center")

        if not center:
            continue

        lat = center["lat"]
        lon = center["lon"]

    distance = geodesic(
        (accident_lat, accident_lon),
        (lat, lon)
    ).km

    hospitals.append({
        "name": name,
        "distance": distance,
        "lat": lat,
        "lon": lon
    })


# Sort by distance
hospitals.sort(key=lambda x: x["distance"])

print("\n🏥 NEAREST HOSPITALS")
print("====================")

for hospital in hospitals[:5]:
    print(
        f"{hospital['name']} - "
        f"{hospital['distance']:.2f} km"
    )


if hospitals:

    nearest = hospitals[0]

    print("\n🚨 SELECTED NEAREST HOSPITAL")
    print("============================")
    print("Hospital:", nearest["name"])
    print("Distance:", f"{nearest['distance']:.2f} km")
    print(
        "Coordinates:",
        nearest["lat"],
        nearest["lon"]
    )

else:
    print("No hospitals found.")