import streamlit as st

from ultralytics import YOLO

import cv2
import numpy as np

import tempfile

import os

import subprocess

import imageio_ffmpeg

import requests

from geopy.distance import geodesic

from collections import defaultdict

import smtplib

from email.message import EmailMessage

import pandas as pd
import sqlite3

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image






# =========================================================

# CONFIGURATION

# =========================================================
# Track occupied ambulances
if "busy_ambulances" not in st.session_state:
    st.session_state.busy_ambulances = set()


def get_available_ambulance(ambulances, accident_location):

    available = [
        ambulance for ambulance in ambulances
        if ambulance["name"] not in st.session_state.busy_ambulances
    ]

    if not available:
        return None

    # Select the nearest available ambulance
    nearest = min(
        available,
        key=lambda ambulance: calculate_distance(
            accident_location,
            (ambulance["latitude"], ambulance["longitude"])
        )
    )

    # Mark selected ambulance as occupied
    st.session_state.busy_ambulances.add(nearest["name"])

    return nearest


SENDER_EMAIL = "nayanad2765@gmail.com"

RECEIVER_EMAIL = "nayanad2765@gmail.com"



# Use the SAME 16-character Google App Password

APP_PASSWORD = os.getenv("ROADGUARD_APP_PASSWORD", "")



# Demo accident location

ACCIDENT_LAT = 12.9716

ACCIDENT_LON = 77.5946

# DEMO ONLY: predefined ambulance coordinates and demo recipient.
# Replace with verified service data before any real-world deployment.
AMBULANCES = [
    {"ambulance_id": "AMB-101", "name": "Ambulance 101", "lat": 12.9680, "lon": 77.6000, "status": "Available", "email": RECEIVER_EMAIL},
    {"ambulance_id": "AMB-102", "name": "Ambulance 102", "lat": 12.9820, "lon": 77.5850, "status": "Available", "email": RECEIVER_EMAIL},
    {"ambulance_id": "AMB-103", "name": "Ambulance 103", "lat": 12.9550, "lon": 77.6100, "status": "Busy", "email": RECEIVER_EMAIL},
]



# Hospital database

HOSPITAL_FILE = "Dataset/hospital.csv"





# =========================================================

# PAGE CONFIG

# =========================================================



st.set_page_config(

    page_title="RoadGuard AI",

    page_icon="🚨",

    layout="wide"

)





# =========================================================

# HOSPITAL DATABASE

# =========================================================



try:

    hospitals_df = pd.read_csv(HOSPITAL_FILE)

except Exception:

    hospitals_df = pd.DataFrame(

        columns=["hospital_name", "email"]

    )





# =========================================================

# CUSTOM CSS

# =========================================================



st.markdown("""

<style>



.stApp {

    background-color: #0b0f14;

    color: white;

}



section[data-testid="stSidebar"] {

    background-color: #111820;

}



.main-title {

    font-size: 38px;

    font-weight: 700;

    margin-bottom: 5px;

}



.subtitle {

    color: #8b98a8;

    font-size: 16px;

    margin-bottom: 25px;

}



.card {

    background-color: #151c24;

    border: 1px solid #26313d;

    border-radius: 12px;

    padding: 20px;

    margin-bottom: 15px;

}



.card-title {

    color: #8b98a8;

    font-size: 14px;

    margin-bottom: 8px;

}



.card-value {

    color: white;

    font-size: 25px;

    font-weight: 700;

}



.alert-card {

    background-color: #35151a;

    border: 1px solid #8b2635;

    border-radius: 12px;

    padding: 20px;

    margin-top: 20px;

}



.success-card {

    background-color: #10271d;

    border: 1px solid #21613d;

    border-radius: 12px;

    padding: 20px;

    margin-top: 20px;

}



.info-card {

    background-color: #151c24;

    border: 1px solid #26313d;

    border-radius: 12px;

    padding: 20px;

}



</style>

""", unsafe_allow_html=True)





# =========================================================

# LOAD YOLO

# =========================================================



@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


@st.cache_resource
def load_accident_classifier():
    """Load the trained PyTorch MobileNetV2 classifier saved by train_model.py."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "roadguard_accident_model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained classifier not found: {model_path}. "
            "Run train_model.py first and keep the .pth file beside app.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", ["Accident", "Non Accident"])

    classifier = models.mobilenet_v2(weights=None)
    classifier.classifier[1] = nn.Linear(
        classifier.last_channel, len(class_names)
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device)
    classifier.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    return classifier, class_names, preprocess, device


model = None





# =========================================================

# SIDEBAR

# =========================================================



st.sidebar.markdown(

    "<h2>🚨 ROADGUARD AI</h2>",

    unsafe_allow_html=True

)



st.sidebar.markdown("### System Status")



st.sidebar.success("● SYSTEM ONLINE")



page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",


        "Ambulance Dashboard",

        "Hospital Dashboard",

        "System Status"

    ]

)





# =========================================================

# IOU FUNCTION

# =========================================================



def calculate_iou(box1, box2):



    x1 = max(box1[0], box2[0])

    y1 = max(box1[1], box2[1])



    x2 = min(box1[2], box2[2])

    y2 = min(box1[3], box2[3])



    intersection = max(

        0,

        x2 - x1

    ) * max(

        0,

        y2 - y1

    )



    area1 = max(

        0,

        box1[2] - box1[0]

    ) * max(

        0,

        box1[3] - box1[1]

    )



    area2 = max(

        0,

        box2[2] - box2[0]

    ) * max(

        0,

        box2[3] - box2[1]

    )



    union = area1 + area2 - intersection



    if union == 0:

        return 0



    return intersection / union





# =========================================================

# =========================================================
# FIND NEAREST HOSPITAL
# =========================================================

def find_nearest_hospital(lat, lon):

    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:5000,{lat},{lon});
      way["amenity"="hospital"](around:5000,{lat},{lon});
      relation["amenity"="hospital"](around:5000,{lat},{lon});
    );
    out center;
    """

    urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]

    headers = {
        "User-Agent": "RoadGuardAI/1.0"
    }

    for url in urls:

        try:

            response = requests.post(
                url,
                data={"data": query},
                headers=headers,
                timeout=30
            )

            print(
                "Hospital API status:",
                response.status_code
            )

            if response.status_code != 200:
                continue

            data = response.json()

            hospitals = []

            for place in data.get("elements", []):

                tags = place.get("tags", {})

                name = tags.get(
                    "name",
                    "Unnamed Hospital"
                )

                if place["type"] == "node":

                    hospital_lat = place["lat"]
                    hospital_lon = place["lon"]

                else:

                    center = place.get("center")

                    if not center:
                        continue

                    hospital_lat = center["lat"]
                    hospital_lon = center["lon"]

                distance = geodesic(
                    (lat, lon),
                    (hospital_lat, hospital_lon)
                ).km

                hospitals.append({
                    "name": name,
                    "distance": distance,
                    "lat": hospital_lat,
                    "lon": hospital_lon
                })

            if hospitals:

                hospitals.sort(
                    key=lambda x: x["distance"]
                )

                print(
                    "Nearest hospital:",
                    hospitals[0]["name"]
                )

                return hospitals[0]

        except Exception as e:

            print(
                "Hospital server failed:",
                e
            )

    # -----------------------------------------------------
    # DEMO FALLBACK
    # -----------------------------------------------------

    print(
        "Using verified demo hospital fallback."
    )

    return {
        "name": "Vydehi Superspeciality Hospital",
        "distance": 0.38,
        "lat": 12.9681553,
        "lon": 77.5950421
    }





# =========================================================

# GET HOSPITAL EMAIL

# =========================================================



def get_hospital_email(hospital_name):



    try:



        for _, row in hospitals_df.iterrows():



            csv_name = str(

                row["hospital_name"]

            ).strip()



            if (

                csv_name.lower()

                ==

                hospital_name.strip().lower()

            ):



                email = str(

                    row["email"]

                ).strip()



                if (

                    email

                    and email.lower() != "nan"

                ):



                    return email



    except Exception as e:



        print(

            "Hospital CSV error:",

            e

        )



    return None





# =========================================================

# SEND EMAIL ALERT

# =========================================================



def send_email_alert(

    severity,

    hospital_name,

    hospital_distance,

    recipient_email

):



    msg = EmailMessage()



    msg["Subject"] = (

        "🚨 ROADGUARD AI - "

        f"{severity} ACCIDENT ALERT"

    )



    msg["From"] = SENDER_EMAIL



    msg["To"] = recipient_email



    email_body = f"""

ROADGUARD AI

ROAD ACCIDENT ALERT

================================



🚨 ACCIDENT DETECTED



Severity:

{severity}



📍 Accident Location:

Bengaluru



Latitude:

{ACCIDENT_LAT}



Longitude:

{ACCIDENT_LON}



🏥 Nearest Hospital:

{hospital_name}



📏 Hospital Distance:

{hospital_distance:.2f} km



Location:

https://www.google.com/maps?q={ACCIDENT_LAT},{ACCIDENT_LON}



================================



RoadGuard AI

Cloud-Based Road Accident Detection

and Automated Emergency Alert System

"""



    msg.set_content(

        email_body

    )



    if not APP_PASSWORD:
        print("EMAIL ERROR: Set ROADGUARD_APP_PASSWORD in the environment.")
        return False

    try:



        with smtplib.SMTP_SSL(

            "smtp.gmail.com",

            465

        ) as server:



            server.login(

                SENDER_EMAIL,

                APP_PASSWORD

            )



            server.send_message(

                msg

            )



        print(

            "EMAIL SENT SUCCESSFULLY"

        )



        return True



    except Exception as e:



        print(

            "EMAIL ERROR:",

            e

        )



        return False





# =========================================================

# AMBULANCE HELPERS

def get_ambulance_records():
    """Load demo ambulance records from SQLite; fall back to in-file demo data."""
    try:
        with sqlite3.connect("roadguard.db") as conn:
            rows = conn.execute("""
                SELECT ambulance_id, name, email, latitude, longitude, status
                FROM ambulances
            """).fetchall()
        if rows:
            return [
                {
                    "ambulance_id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "lat": row[3],
                    "lon": row[4],
                    "status": row[5],
                }
                for row in rows
            ]
    except (sqlite3.Error, OSError):
        pass
    return [dict(item) for item in AMBULANCES]


def find_nearest_ambulance(lat, lon):
    available = [
        dict(ambulance) for ambulance in get_ambulance_records()
        if str(ambulance.get("status", "")).lower() == "available"
        and ambulance.get("lat") is not None
        and ambulance.get("lon") is not None
    ]
    if not available:
        return None
    for ambulance in available:
        ambulance["distance"] = geodesic(
            (lat, lon), (ambulance["lat"], ambulance["lon"])
        ).km
    return min(available, key=lambda ambulance: ambulance["distance"])


def send_ambulance_alert(ambulance, severity, hospital=None):
    """Send a demo alert to the selected ambulance contact."""
    recipient_email = ambulance.get("email")
    # Demo addresses ending in example.com are not deliverable.
    if not recipient_email or recipient_email.lower().endswith("@example.com"):
        recipient_email = os.getenv("ROADGUARD_TEST_RECIPIENT", RECEIVER_EMAIL)

    if not recipient_email or not APP_PASSWORD:
        print("AMBULANCE EMAIL: Configure ROADGUARD_APP_PASSWORD and a valid recipient.")
        return False

    accident_map = f"https://www.google.com/maps/search/?api=1&query={ACCIDENT_LAT},{ACCIDENT_LON}"
    hospital_name = hospital.get("name", "Not available") if hospital else "Not available"
    hospital_distance = hospital.get("distance", None) if hospital else None
    hospital_map = None
    if hospital and hospital.get("lat") is not None and hospital.get("lon") is not None:
        hospital_map = f"https://www.google.com/maps/search/?api=1&query={hospital['lat']},{hospital['lon']}"

    hospital_distance_text = (
        f"{hospital_distance:.2f} km (straight-line distance)"
        if isinstance(hospital_distance, (int, float)) else "Not available"
    )

    msg = EmailMessage()
    msg["Subject"] = f"[AMBULANCE ACTION] RoadGuard AI - {severity} accident"
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg.set_content(f"""ROADGUARD AI — AMBULANCE ALERT (DEMO)

ACTION: Review the accident location first.

ACCIDENT LOCATION
Place: Bengaluru (predefined demo location)
Coordinates: {ACCIDENT_LAT}, {ACCIDENT_LON}
Open accident location in Google Maps:
{accident_map}

NEAREST HOSPITAL
Hospital: {hospital_name}
Straight-line distance: {hospital_distance_text}
Hospital directions:
{hospital_map or 'Not available'}

INCIDENT
Severity estimate: {severity}
Assigned ambulance: {ambulance.get('name', 'Ambulance')} ({ambulance.get('ambulance_id', 'ID unavailable')})
Estimated straight-line distance to accident: {ambulance.get('distance', 0):.2f} km

DEMO ONLY: Coordinates and ambulance availability are predefined. This message
is not a real dispatch or confirmation from an emergency service.
""")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as exc:
        print("AMBULANCE EMAIL ERROR:", exc)
        return False


def send_hospital_alert(severity, hospital, ambulance, recipient_email):
    """Send the hospital an incident alert with ambulance information."""
    if not recipient_email or not APP_PASSWORD:
        print("HOSPITAL EMAIL: Configure ROADGUARD_APP_PASSWORD and a valid recipient.")
        return False

    accident_map = f"https://www.google.com/maps/search/?api=1&query={ACCIDENT_LAT},{ACCIDENT_LON}"
    hospital_map = ""
    if hospital and hospital.get("lat") is not None and hospital.get("lon") is not None:
        hospital_map = f"https://www.google.com/maps/search/?api=1&query={hospital['lat']},{hospital['lon']}"

    ambulance_details = "No available ambulance identified."
    if ambulance:
        ambulance_details = (
            f"{ambulance.get('name', 'Ambulance')} ({ambulance.get('ambulance_id', 'ID unavailable')})\n"
            f"Distance to accident: {ambulance.get('distance', 0):.2f} km\n"
            f"Contact email: {ambulance.get('email', 'Not configured')}"
        )

    msg = EmailMessage()
    msg["Subject"] = f"[HOSPITAL NOTIFICATION] RoadGuard AI - {severity} incident"
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg.set_content(f"""ROADGUARD AI — HOSPITAL NOTIFICATION (DEMO)

INCIDENT SUMMARY
Severity estimate: {severity}
Place: Bengaluru (predefined demo location)
Coordinates: {ACCIDENT_LAT}, {ACCIDENT_LON}
Accident location in Google Maps:
{accident_map}

HOSPITAL DETAILS
Nearest hospital identified: {hospital.get('name', 'Not available') if hospital else 'Not available'}
Hospital map:
{hospital_map or 'Not available'}

AMBULANCE INFORMATION
{ambulance_details}

Please note: this is a prototype notification for testing only. Coordinates are
predefined, and no real hospital or ambulance service is being contacted or dispatched.
""")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as exc:
        print("HOSPITAL EMAIL ERROR:", exc)
        return False




# =========================================================
# SHARED INCIDENT HISTORY (SQLite)
# =========================================================
def initialize_incident_history():
    with sqlite3.connect("roadguard.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT UNIQUE,
                place TEXT,
                latitude REAL,
                longitude REAL,
                severity TEXT,
                ambulance_id TEXT,
                ambulance_name TEXT,
                hospital_name TEXT,
                hospital_latitude REAL,
                hospital_longitude REAL,
                ambulance_email_sent INTEGER DEFAULT 0,
                hospital_email_sent INTEGER DEFAULT 0,
                response_status TEXT DEFAULT 'Awaiting acknowledgement',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_incident(video_name, severity, ambulance, hospital,
                  ambulance_email_sent=False, hospital_email_sent=False):
    initialize_incident_history()
    with sqlite3.connect("roadguard.db") as conn:
        conn.execute("""
            INSERT INTO incidents (
                video_name, place, latitude, longitude, severity,
                ambulance_id, ambulance_name, hospital_name,
                hospital_latitude, hospital_longitude,
                ambulance_email_sent, hospital_email_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_name) DO UPDATE SET
                place=excluded.place,
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                severity=excluded.severity,
                ambulance_id=excluded.ambulance_id,
                ambulance_name=excluded.ambulance_name,
                hospital_name=excluded.hospital_name,
                hospital_latitude=excluded.hospital_latitude,
                hospital_longitude=excluded.hospital_longitude,
                ambulance_email_sent=MAX(incidents.ambulance_email_sent, excluded.ambulance_email_sent),
                hospital_email_sent=MAX(incidents.hospital_email_sent, excluded.hospital_email_sent)
        """, (
            video_name, "Bengaluru (predefined demo location)",
            ACCIDENT_LAT, ACCIDENT_LON, severity,
            ambulance.get("ambulance_id") if ambulance else None,
            ambulance.get("name") if ambulance else None,
            hospital.get("name") if hospital else None,
            hospital.get("lat") if hospital else None,
            hospital.get("lon") if hospital else None,
            int(bool(ambulance_email_sent)), int(bool(hospital_email_sent))
        ))

def load_incidents(limit=50):
    initialize_incident_history()
    with sqlite3.connect("roadguard.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY incident_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]

initialize_incident_history()

# DASHBOARD

# =========================================================



if page == "Dashboard":



    st.markdown(

        '<div class="main-title">'

        'Road Accident Detection'

        '</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        '<div class="subtitle">'

        'AI-powered accident detection and emergency response system'

        '</div>',

        unsafe_allow_html=True

    )



    uploaded_file = st.file_uploader(

        "Upload Traffic Video",

        type=[

            "mp4",

            "avi",

            "mov"

        ]

    )



    if uploaded_file:



        # -------------------------------------------------

        # RESET ALERT FOR NEW VIDEO

        # -------------------------------------------------



        if (

            "last_uploaded_file"

            not in st.session_state

            or

            st.session_state.last_uploaded_file

            != uploaded_file.name

        ):



            st.session_state.alert_sent = False
            st.session_state.ambulance_alert_sent = False



            st.session_state.last_uploaded_file = (

                uploaded_file.name

            )



        # -------------------------------------------------

        # SAVE INPUT VIDEO

        # -------------------------------------------------



        temp_input = tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".mp4"

        )



        temp_input.write(

            uploaded_file.read()

        )



        temp_input.close()



        st.subheader(

            "🎥 Original Video"

        )



        st.video(

            temp_input.name

        )



        # -------------------------------------------------

        # OUTPUT FILES

        # -------------------------------------------------



        output_avi = tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".avi"

        ).name



        output_mp4 = tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".mp4"

        ).name



        cap = cv2.VideoCapture(

            temp_input.name

        )



        fps = cap.get(

            cv2.CAP_PROP_FPS

        )



        width = int(

            cap.get(

                cv2.CAP_PROP_FRAME_WIDTH

            )

        )



        height = int(

            cap.get(

                cv2.CAP_PROP_FRAME_HEIGHT

            )

        )



        if fps <= 0:

            fps = 25



        out = cv2.VideoWriter(

            output_avi,

            cv2.VideoWriter_fourcc(

                *"MJPG"

            ),

            fps,

            (

                width,

                height

            )

        )



        # -------------------------------------------------

        # DETECTION VARIABLES

        # -------------------------------------------------



        vehicle_classes = {

            "car",

            "motorcycle",

            "bus",

            "truck"

        }



        previous_positions = {}



        collision_counter = defaultdict(

            int

        )



        accident_detected = False
        frame_count = 0
        model = load_model()
        classifier, classifier_classes, classifier_preprocess, classifier_device = (
            load_accident_classifier()
        )
        accident_votes = 0
        non_accident_votes = 0
        latest_prediction = "Waiting for frame classification"


        severity = "NO ACCIDENT"



        vehicle_ids = set()



        # -------------------------------------------------

        # PROCESS VIDEO

        # -------------------------------------------------



        while True:



            ret, frame = cap.read()



            if not ret:
                break

            frame_count += 1

            # Sample more frequently so a brief collision is less likely to be missed.
            # DEMO-SENSITIVE RULE: once any sampled frame is classified as Accident,
            # keep the accident flag active for the remainder of this uploaded video.
            if frame_count % 2 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                input_tensor = classifier_preprocess(pil_frame).unsqueeze(0).to(
                    classifier_device
                )
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    predicted_index = int(torch.argmax(probabilities).item())
                latest_prediction = classifier_classes[predicted_index]

                if latest_prediction.strip().lower().replace("-", " ") == "accident":
                    accident_votes += 1
                    accident_detected = True
                else:
                    non_accident_votes += 1

            results = model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=0.5,

                verbose=False

            )



            vehicles = []



            boxes = results[0].boxes



            if boxes.id is not None:



                ids = (

                    boxes.id

                    .cpu()

                    .numpy()

                    .astype(int)

                )



                coords = (

                    boxes.xyxy

                    .cpu()

                    .numpy()

                )



                classes = (

                    boxes.cls

                    .cpu()

                    .numpy()

                    .astype(int)

                )



                for (

                    box,

                    track_id,

                    cls

                ) in zip(

                    coords,

                    ids,

                    classes

                ):



                    label = model.names[cls]



                    if label in vehicle_classes:



                        x1, y1, x2, y2 = map(

                            int,

                            box

                        )



                        cx = (

                            x1 + x2

                        ) // 2



                        cy = (

                            y1 + y2

                        ) // 2



                        vehicles.append({

                            "id": track_id,

                            "box": box,

                            "center": (

                                cx,

                                cy

                            ),

                            "label": label

                        })



                        vehicle_ids.add(

                            track_id

                        )



                        # Bounding box



                        cv2.rectangle(

                            frame,

                            (x1, y1),

                            (x2, y2),

                            (255, 0, 255),

                            2

                        )



                        cv2.putText(

                            frame,

                            f"ID:{track_id} {label}",

                            (

                                x1,

                                max(

                                    y1 - 10,

                                    20

                                )

                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.6,

                            (255, 0, 255),

                            2

                        )



            # -------------------------------------------------

            # MOVEMENT

            # -------------------------------------------------



            movements = []



            for vehicle in vehicles:



                vid = vehicle["id"]



                center = vehicle["center"]



                if vid in previous_positions:



                    old_center = (

                        previous_positions[vid]

                    )



                    movement = (

                        (

                            center[0]

                            - old_center[0]

                        ) ** 2

                        +

                        (

                            center[1]

                            - old_center[1]

                        ) ** 2

                    ) ** 0.5



                    movements.append(

                        movement

                    )



                previous_positions[vid] = center



            # -------------------------------------------------

            # COLLISION DETECTION

            # -------------------------------------------------



            for i in range(

                len(vehicles)

            ):



                for j in range(

                    i + 1,

                    len(vehicles)

                ):



                    v1 = vehicles[i]

                    v2 = vehicles[j]



                    pair = tuple(

                        sorted(

                            (

                                v1["id"],

                                v2["id"]

                            )

                        )

                    )



                    overlap = calculate_iou(

                        v1["box"],

                        v2["box"]

                    )



                    if overlap > 0.08:



                        collision_counter[

                            pair

                        ] += 1



                    else:



                        collision_counter[

                            pair

                        ] = 0



                    # Vehicle overlap is retained for tracking analysis only.
                    # Accident status is determined by the trained frame classifier.



            # -------------------------------------------------

            # SEVERITY

            # -------------------------------------------------



            if accident_detected:



                average_movement = (

                    sum(movements)

                    /

                    len(movements)

                    if movements

                    else 0

                )



                if average_movement > 15:



                    severity = "MINOR"



                elif average_movement > 5:



                    severity = "MODERATE"



                else:



                    severity = "MAJOR"



            # -------------------------------------------------

            # VIDEO STATUS

            # -------------------------------------------------



            if accident_detected:



                cv2.putText(

                    frame,

                    "ACCIDENT DETECTED",

                    (30, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.0,

                    (0, 0, 255),

                    3

                )



                cv2.putText(

                    frame,

                    f"Severity: {severity}",

                    (30, 90),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.9,

                    (0, 0, 255),

                    3

                )



            else:



                cv2.putText(

                    frame,

                    "NO ACCIDENT",

                    (30, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.0,

                    (0, 255, 0),

                    3

                )



            cv2.putText(

                frame,

                "ROADGUARD AI",

                (

                    30,

                    height - 25

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )



            out.write(

                frame

            )



        # Severity is a simple movement-based demo estimate, not a trained model.
        if not accident_detected:
            severity = "NO ACCIDENT"

        cap.release()



        out.release()



        # -------------------------------------------------

        # CONVERT AVI TO MP4

        # -------------------------------------------------



        ffmpeg_path = (

            imageio_ffmpeg.get_ffmpeg_exe()

        )



        subprocess.run(

            [

                ffmpeg_path,

                "-y",

                "-i",

                output_avi,

                "-c:v",

                "libx264",

                "-pix_fmt",

                "yuv420p",

                "-movflags",

                "+faststart",

                output_mp4

            ],

            stdout=subprocess.DEVNULL,

            stderr=subprocess.DEVNULL

        )



        # -------------------------------------------------

        # PROCESSED VIDEO

        # -------------------------------------------------



        st.subheader(

            "🤖 AI Processed Video"

        )



        with open(

            output_mp4,

            "rb"

        ) as video_file:



            video_bytes = (

                video_file.read()

            )



        st.video(

            video_bytes,

            format="video/mp4"

        )



        st.caption(
            f"Trained image classifier sampled {accident_votes + non_accident_votes} frames: "
            f"{accident_votes} Accident / {non_accident_votes} Non Accident votes. "
            "Frame classification can still be wrong; this prototype is not for real-world emergency dispatch."
        )

        st.download_button(

            "Download Processed Video",

            video_bytes,

            file_name="roadguard_processed.mp4",

            mime="video/mp4"

        )



        # -------------------------------------------------

        # HOSPITAL SEARCH

        # -------------------------------------------------



        hospital = None



        if accident_detected:



            with st.spinner(

                "Finding nearest hospital..."

            ):



                hospital = (

                    find_nearest_hospital(

                        ACCIDENT_LAT,

                        ACCIDENT_LON

                    )

                )



        # -------------------------------------------------

        # NEAREST AVAILABLE AMBULANCE

        ambulance = None
        if accident_detected:
            ambulance = find_nearest_ambulance(ACCIDENT_LAT, ACCIDENT_LON)

        ambulance_email_sent = False
        if accident_detected and ambulance and not st.session_state.get("ambulance_alert_sent", False):
            ambulance_email_sent = send_ambulance_alert(
                ambulance, severity, hospital
            )
            if ambulance_email_sent:
                st.session_state.ambulance_alert_sent = True

        # HOSPITAL EMAIL

        # -------------------------------------------------



        hospital_email = None



        if hospital:



            hospital_email = (

                get_hospital_email(

                    hospital["name"]

                )

            )



            # For current demo/testing,

            # use project receiver if hospital

            # email is not configured.

            if not hospital_email:



                hospital_email = (

                    RECEIVER_EMAIL

                )



        # -------------------------------------------------

        # SEND EMAIL ALERT

        # -------------------------------------------------



        email_sent = False



        if (

            accident_detected

            and hospital

            and hospital_email

        ):



            if not st.session_state.get(

                "alert_sent",

                False

            ):



                with st.spinner(

                    "Sending emergency alert..."

                ):



                    email_sent = (

                        send_hospital_alert(
                            severity,
                            hospital,
                            ambulance,
                            hospital_email
                        )

                    )



                if email_sent:



                    st.session_state.alert_sent = True



        # -------------------------------------------------

        if accident_detected:
            save_incident(
                uploaded_file.name,
                severity,
                ambulance,
                hospital,
                ambulance_email_sent=(ambulance_email_sent or st.session_state.get("ambulance_alert_sent", False)),
                hospital_email_sent=(email_sent or st.session_state.get("alert_sent", False))
            )

        # DASHBOARD CARDS

        # -------------------------------------------------



        st.html(

            "<br>",



        )



        col1, col2, col3, col4 = (

            st.columns(4)

        )



        with col1:



            st.html(

                f"""

                <div class="card">

                    <div class="card-title">

                        UNIQUE VEHICLES

                    </div>



                    <div class="card-value">

                        {len(vehicle_ids)}

                    </div>

                </div>

                """,



            )



        with col2:



            status = (

                "DETECTED"

                if accident_detected

                else "CLEAR"

            )



            st.html(

                f"""

                <div class="card">

                    <div class="card-title">

                        ACCIDENT STATUS

                    </div>



                    <div class="card-value">

                        {status}

                    </div>

                </div>

                """,



            )



        with col3:



            st.html(

                """

                <div class="card">

                    <div class="card-title">

                        TRACKING

                    </div>



                    <div class="card-value">

                        ACTIVE

                    </div>

                </div>

                """,



            )



        with col4:



            st.html(

                f"""

                <div class="card">

                    <div class="card-title">

                        SEVERITY

                    </div>



                    <div class="card-value">

                        {severity}

                    </div>

                </div>

                """,



            )



        # =================================================

        # INCIDENT INFORMATION

        # =================================================



        st.subheader(

            "📍 Incident Information"

        )



        col1, col2 = st.columns(2)



        # -------------------------------------------------

        # LOCATION CARD

        # -------------------------------------------------



        with col1:



            st.html(

                f"""

                <div class="info-card">



                    <div class="card-title">

                        📍 ACCIDENT LOCATION

                    </div>



                    <div class="card-value">

                        Bengaluru

                    </div>



                    <p>

                        <b>Latitude:</b>

                        {ACCIDENT_LAT}

                        <br>



                        <b>Longitude:</b>

                        {ACCIDENT_LON}

                    </p>



                </div>

                """,



            )



        # -------------------------------------------------

        # HOSPITAL CARD

        # -------------------------------------------------



        with col2:



            if hospital:



                hospital_name = (

                    hospital["name"]

                )



                distance = (

                    hospital["distance"]

                )



                hospital_lat = (

                    hospital["lat"]

                )



                hospital_lon = (

                    hospital["lon"]

                )



            else:



                hospital_name = (

                    "Not Available"

                )



                distance = 0



                hospital_lat = 0



                hospital_lon = 0



            display_email = (

                hospital_email

                if hospital_email

                else "Not Configured"

            )



            st.html(

                f"""

                <div class="info-card">



                    <div class="card-title">

                        🏥 NEAREST HOSPITAL

                    </div>



                    <div class="card-value">

                        {hospital_name}

                    </div>



                    <p>

                        <b>Distance:</b>

                        {distance:.2f} km

                        <br>



                        <b>Emergency Email:</b>

                        {display_email}

                        <br>



                        <b>Hospital Latitude:</b>

                        {hospital_lat}

                        <br>



                        <b>Hospital Longitude:</b>

                        {hospital_lon}

                    </p>



                </div>

                """,



            )



        # =================================================

        # AMBULANCE INFORMATION

        st.subheader("🚑 Nearest Available Ambulance")
        if accident_detected and ambulance:
            st.success(
                f"{ambulance['name']} ({ambulance['ambulance_id']}) — "
                f"{ambulance['distance']:.2f} km away"
            )
            st.write(f"Status: {ambulance['status']}")
            st.link_button(
                "Open accident location in Google Maps",
                f"https://www.google.com/maps?q={ACCIDENT_LAT},{ACCIDENT_LON}"
            )
            if ambulance_email_sent or st.session_state.get("ambulance_alert_sent", False):
                st.caption("Ambulance alert email sent successfully.")
            else:
                st.caption("Ambulance email not confirmed. Check terminal output and email configuration.")
        elif accident_detected:
            st.warning("No available ambulance is registered in the demo data.")
        else:
            st.info("An ambulance will be selected when an accident is detected.")


# =========================================================

# SYSTEM STATUS

# =========================================================



elif page == "System Status":



    st.markdown(

        '<div class="main-title">'

        'System Status'

        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'RoadGuard AI system components'
        '</div>',
        unsafe_allow_html=True
    )



    st.success(

        "🟢 YOLO Detection — Online"

    )



    st.success(

        "🟢 ByteTrack Tracking — Online"

    )



    st.success(

        "🟢 Accident Detection — Online"

    )



    st.success(

        "🟢 Severity Analysis — Online"

    )



    st.success(

        "🟢 Hospital Detection — Online"

    )



    st.success(

        "🟢 Email Alert System — Online"

    )

# =========================================================
# AMBULANCE DASHBOARD
# =========================================================

elif page == "Ambulance Dashboard":
    st.markdown('<div class="main-title">Ambulance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predefined ambulance locations and availability</div>', unsafe_allow_html=True)

    ambulance_df = pd.DataFrame(get_ambulance_records())
    ambulance_df["distance_from_demo_accident_km"] = ambulance_df.apply(
        lambda row: geodesic(
            (ACCIDENT_LAT, ACCIDENT_LON), (row["lat"], row["lon"])
        ).km,
        axis=1
    )

    st.dataframe(
        ambulance_df[["ambulance_id", "name", "status", "distance_from_demo_accident_km"]]
        .round({"distance_from_demo_accident_km": 2}),
        use_container_width=True,
        hide_index=True
    )
    st.subheader("📍 Ambulance Locations")
    st.map(
        ambulance_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
        latitude="latitude",
        longitude="longitude",
        size=100
    )

    # Show the accident location FIRST, followed by the nearest hospital.
    st.divider()
    st.subheader("🚨 Accident Location")
    st.write("**Place:** Bengaluru (predefined demo location)")
    st.write(f"**Latitude:** {ACCIDENT_LAT}")
    st.write(f"**Longitude:** {ACCIDENT_LON}")
    st.link_button(
        "📍 Open Accident Location in Google Maps",
        f"https://www.google.com/maps/dir/?api=1&destination={ACCIDENT_LAT},{ACCIDENT_LON}"
    )

    st.divider()
    st.subheader("🏥 Nearest Hospital")
    with st.spinner("Finding the nearest hospital..."):
        nearest_hospital = find_nearest_hospital(ACCIDENT_LAT, ACCIDENT_LON)

    if nearest_hospital:
        st.markdown(f"### {nearest_hospital['name']}")
        st.write(f"**Distance from accident:** {nearest_hospital['distance']:.2f} km (straight-line estimate)")
        st.link_button(
            "🗺️ Get Directions to Nearest Hospital",
            f"https://www.google.com/maps/dir/?api=1&destination={nearest_hospital['lat']},{nearest_hospital['lon']}"
        )
        st.caption(
            "Google Maps will calculate a driving route. The displayed distance is "
            "straight-line distance, not road distance or ETA."
        )
    else:
        st.warning("Nearest hospital could not be found. Please try again later.")

    st.subheader("🚨 Latest Accident Details")
    incidents = load_incidents()
    if incidents:
        latest = incidents[0]
        st.write(f"**Incident:** RG-{latest['incident_id']:05d}")
        st.write(f"**Severity:** {latest['severity']}")
        st.write(f"**Location:** {latest['place']}")
        st.link_button(
            "📍 Open accident location",
            f"https://www.google.com/maps/dir/?api=1&destination={latest['latitude']},{latest['longitude']}"
        )
        st.write(f"**Assigned ambulance:** {latest['ambulance_name'] or 'Not assigned'}")
        st.write(f"**Ambulance ID:** {latest['ambulance_id'] or 'Not available'}")
        st.write(
            f"**Ambulance email status:** "
            f"{'Sent' if latest['ambulance_email_sent'] else 'Not confirmed / not sent'}"
        )
        st.write(f"**Nearest hospital:** {latest['hospital_name'] or 'Not available'}")
        if latest["hospital_latitude"] is not None and latest["hospital_longitude"] is not None:
            st.link_button(
                "🏥 Get directions to nearest hospital",
                f"https://www.google.com/maps/dir/?api=1&destination={latest['hospital_latitude']},{latest['hospital_longitude']}"
            )
        st.write(
            f"**Hospital email status:** "
            f"{'Sent' if latest['hospital_email_sent'] else 'Not confirmed / not sent'}"
        )
        st.write(f"**Response status:** {latest['response_status']}")
        st.subheader("📋 Recent Incident History")
        st.dataframe(pd.DataFrame(incidents)[
            ["incident_id", "severity", "place", "ambulance_name", "hospital_name",
             "response_status", "created_at"]
        ], use_container_width=True, hide_index=True)
    else:
        st.info("No saved incidents yet. Upload a video on the main Dashboard first.")

    st.caption(
        "Demo only: accident and ambulance coordinates are predefined. Hospital results "
        "depend on map data. Verify locations and contacts before real-world use."
    )


# =========================================================
# HOSPITAL DASHBOARD
# =========================================================

elif page == "Hospital Dashboard":
    st.markdown('<div class="main-title">🏥 Hospital Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Demo view of accident alerts received by a hospital</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This is a prototype dashboard. It displays the configured demo incident; "
        "it does not yet receive live alerts from a real hospital system."
    )

    incident_col, severity_col, status_col = st.columns(3)
    with incident_col:
        st.metric("Incident", "RG-DEMO-001")
    with severity_col:
        st.metric("Severity", "Demo / verify")
    with status_col:
        st.metric("Response status", "Awaiting acknowledgement")

    st.subheader("📍 Accident Details")
    st.write("**Location:** Bengaluru (predefined demo coordinates)")
    st.write(f"**Latitude:** {ACCIDENT_LAT}")
    st.write(f"**Longitude:** {ACCIDENT_LON}")
    st.link_button(
        "Open accident location in Google Maps",
        f"https://www.google.com/maps?q={ACCIDENT_LAT},{ACCIDENT_LON}"
    )

    st.subheader("🚑 Ambulance Response Overview")
    ambulance_df = pd.DataFrame(AMBULANCES)
    available_count = int((ambulance_df["status"].str.lower() == "available").sum())
    busy_count = int((ambulance_df["status"].str.lower() == "busy").sum())
    c1, c2 = st.columns(2)
    c1.metric("Available demo ambulances", available_count)
    c2.metric("Busy demo ambulances", busy_count)
    st.dataframe(
        ambulance_df[["ambulance_id", "name", "status"]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📋 Incident History")
    incidents = load_incidents()
    if incidents:
        latest = incidents[0]
        st.success(f"Latest saved incident: RG-{latest['incident_id']:05d}")
        st.write(f"**Severity:** {latest['severity']}")
        st.write(f"**Location:** {latest['place']}")
        st.link_button(
            "📍 Open accident location in Google Maps",
            f"https://www.google.com/maps/dir/?api=1&destination={latest['latitude']},{latest['longitude']}"
        )
        st.write(f"**Assigned ambulance:** {latest['ambulance_name'] or 'Not assigned'}")
        st.write(f"**Nearest hospital:** {latest['hospital_name'] or 'Not available'}")
        st.write(f"**Ambulance email sent:** {'Yes' if latest['ambulance_email_sent'] else 'No / not confirmed'}")
        st.write(f"**Hospital email sent:** {'Yes' if latest['hospital_email_sent'] else 'No / not confirmed'}")
        st.write(f"**Response status:** {latest['response_status']}")
        st.dataframe(pd.DataFrame(incidents)[
            ["incident_id", "severity", "place", "ambulance_name", "hospital_name",
             "ambulance_email_sent", "hospital_email_sent", "response_status", "created_at"]
        ], use_container_width=True, hide_index=True)
    else:
        st.info("No incidents saved yet. Upload a video on the main Dashboard first.")
