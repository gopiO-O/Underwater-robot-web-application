import requests
import json

# Flask server URL
SERVER_URL = "http://localhost:5000"

# This directly calls the server's publish endpoint
# to send {"thres": 1} to SIH/Gopi/pub
# WITHOUT sending any sensor data

try:
    print("📤 Sending threshold message directly to SIH/Gopi/pub...")
    
    response = requests.post(
        f"{SERVER_URL}/publish",
        json={
            "topic": "SIH/Gopi/pub",
            "message": json.dumps({"thres": 1})
        }
    )
    
    result = response.json()
    
    if result.get('ok'):
        print("✅ Message sent successfully!")
        print(f"   Topic: SIH/Gopi/pub")
        print(f'   Message: {{"thres": 1}}')
    else:
        print(f"❌ Failed: {result.get('error')}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure server.py is running on localhost:5000")
