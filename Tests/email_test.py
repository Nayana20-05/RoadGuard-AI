import smtplib
from email.message import EmailMessage

# Your Gmail address
SENDER_EMAIL = "nayanad2765@gmail.com"

# The email where you want to receive the alert
RECEIVER_EMAIL = "nayanad2765@gmail.com"

# Paste your 16-character App Password here
APP_PASSWORD = "ghzq tkvd requ iuaz"

msg = EmailMessage()

msg["Subject"] = "🚨 RoadGuard AI - Accident Alert"
msg["From"] = SENDER_EMAIL
msg["To"] = RECEIVER_EMAIL

msg.set_content("""
🚨 ROAD ACCIDENT ALERT

Accident Status: DETECTED
Severity: MAJOR

Location: Bengaluru
Latitude: 12.9716
Longitude: 77.5946

Nearest Hospital:
Vydehi Superspeciality Hospital

Distance:
0.38 km

RoadGuard AI Emergency Detection System
""")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

    print("✅ EMAIL SENT SUCCESSFULLY!")

except Exception as e:
    print("❌ EMAIL FAILED")
    print(e)