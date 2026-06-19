from flask import Flask, jsonify, request, Response, send_from_directory, send_file
from flask_cors import CORS
import paho.mqtt.client as mqtt
import os
import time
import threading
import json
import cv2
from pathlib import Path
import requests

# Configuration
BROKER = "44.214.52.220"
PORT = 1883
PUBLISH_TOPIC = "SIH/Vishal/pub"  # Topic for sending threshold data
SUBSCRIBE_TOPIC = "SIH/Gopi/pub"  # Subscribe to same topic for sensor data

# Groq API Configuration - PASTE YOUR API KEY HERE
GROQ_API_KEY = "Sorry :)"  
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Google Maps API Key - Keep this secure on the backend
# Prefer setting an environment variable GOOGLE_MAPS_API_KEY; falls back to hardcoded value if not set.
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "Sorry :)")  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Serve only the ./static directory at /static to avoid route conflicts
STATIC_DIR = os.path.join(BASE_DIR, 'static')
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
CORS(app)

# Global state for sensor + location data
latest_sensor_data = {"ph": "--", "tb": "--", "tds": "--"}
sensor_data_history = []
latest_location = {"lat": None, "lon": None, "timestamp": None}
location_history = []  # keep last 200 location points
latest_water_level = {"distance": None}  # ultrasonic sensor distance in cm
subscription_active = False
subscribe_thread = None
mqtt_client = None

# LLM Analysis State
sensor_reading_counter = 0  # Track total readings received
last_llm_analysis = {
    "insight": "Waiting for sensor data... (Analysis triggers every 50 readings)",
    "timestamp": None,
    "ph": None,
    "tb": None,
    "tds": None,
    "reading_count": 0
}


def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker."""
    if rc == 0:
        print(f"[MQTT] Connected to broker, subscribing to {SUBSCRIBE_TOPIC}")
        client.subscribe(SUBSCRIBE_TOPIC)
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")


def on_message(client, userdata, msg):
    """Callback when message received from MQTT. Expected JSON may include lat/lon."""
    global latest_sensor_data, sensor_data_history, latest_location, location_history, latest_water_level, sensor_reading_counter
    try:
        payload = msg.payload.decode(errors='ignore').strip()
        if not payload:
            return
        print(f"[MQTT] Received on {msg.topic}: {payload}")

        data = json.loads(payload)

        # Update water level if ultra field is present (ultrasonic sensor distance)
        if 'ultra' in data:
            latest_water_level['distance'] = float(data['ultra'])
            print(f"[MQTT] Water level updated: {latest_water_level['distance']} cm from sensor")

        # Update sensor values if present
        has_sensor_data = False
        if 'ph' in data:
            latest_sensor_data['ph'] = data['ph']
            has_sensor_data = True
        if 'tb' in data:
            latest_sensor_data['tb'] = data['tb']
            has_sensor_data = True
        if 'tds' in data:
            latest_sensor_data['tds'] = data['tds']
            has_sensor_data = True

        ts = time.strftime('%H:%M:%S')
        sensor_data_entry = {
            'ph': latest_sensor_data['ph'],
            'tb': latest_sensor_data['tb'],
            'tds': latest_sensor_data['tds'],
            'timestamp': ts
        }
        sensor_data_history.append(sensor_data_entry)
        if len(sensor_data_history) > 100:
            sensor_data_history.pop(0)
        
        # Increment counter and trigger LLM every 20 readings
        if has_sensor_data:
            sensor_reading_counter += 1
            print(f"[LLM] Reading count: {sensor_reading_counter}")
            
            if sensor_reading_counter % 20 == 0:
                print(f"[LLM] Triggering analysis at reading #{sensor_reading_counter}")
                # Calculate average of last 20 readings for stable analysis
                recent_count = min(20, len(sensor_data_history))
                recent_data = sensor_data_history[-recent_count:]
                
                # Filter out non-numeric values and calculate averages
                valid_ph = [r['ph'] for r in recent_data if isinstance(r['ph'], (int, float))]
                valid_tb = [r['tb'] for r in recent_data if isinstance(r['tb'], (int, float))]
                valid_tds = [r['tds'] for r in recent_data if isinstance(r['tds'], (int, float))]
                
                if valid_ph and valid_tb and valid_tds:
                    avg_ph = sum(valid_ph) / len(valid_ph)
                    avg_tb = sum(valid_tb) / len(valid_tb)
                    avg_tds = sum(valid_tds) / len(valid_tds)
                    
                    # Run LLM analysis in background thread (non-blocking)
                    threading.Thread(
                        target=get_llm_analysis_async,
                        args=(avg_ph, avg_tb, avg_tds, sensor_reading_counter),
                        daemon=True
                    ).start()

        # Optional location update
        if 'lat' in data and 'lon' in data:
            try:
                lat = float(data['lat'])
                lon = float(data['lon'])
                latest_location = {"lat": lat, "lon": lon, "timestamp": ts}
                location_history.append(latest_location)
                if len(location_history) > 200:
                    location_history.pop(0)
            except (ValueError, TypeError):
                print("[MQTT] Invalid lat/lon values, skipped")

    except json.JSONDecodeError as e:
        print(f"[MQTT] Invalid JSON: {e}")
    except Exception as e:
        print(f"[MQTT] Error processing message: {e}")


def start_mqtt_subscription():
    """Start MQTT subscription in background."""
    global mqtt_client, subscription_active
    
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        mqtt_client.connect(BROKER, PORT, keepalive=60)
        mqtt_client.loop_start()
        subscription_active = True
        print(f"[MQTT] Subscription started on {SUBSCRIBE_TOPIC}")
    except Exception as e:
        print(f"[MQTT] Subscription error: {e}")


@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Start subscribing to sensor data."""
    global subscription_active
    
    if subscription_active:
        return jsonify({'ok': True, 'message': 'Already subscribed'})
    
    start_mqtt_subscription()
    return jsonify({'ok': True, 'message': 'Subscription started'})


@app.route('/unsubscribe', methods=['POST'])
def unsubscribe():
    """Stop subscribing to sensor data."""
    global subscription_active, mqtt_client
    
    if not subscription_active:
        return jsonify({'ok': False, 'error': 'Not subscribed'}), 400
    
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    
    subscription_active = False
    return jsonify({'ok': True, 'message': 'Unsubscribed'})


def get_llm_analysis_async(ph, tb, tds, reading_count):
    """Run Groq LLM analysis in background thread."""
    global last_llm_analysis
    
    try:
        print(f"[LLM] Starting analysis for reading #{reading_count}...")
        insight = analyze_water_quality_groq(ph, tb, tds)
        
        last_llm_analysis = {
            "insight": insight,
            "timestamp": time.strftime('%H:%M:%S'),
            "ph": round(ph, 2),
            "tb": round(tb, 2),
            "tds": round(tds, 2),
            "reading_count": reading_count
        }
        print(f"[LLM] ✓ Analysis complete: {insight[:80]}...")
        
    except Exception as e:
        error_msg = f"LLM analysis failed: {str(e)}"
        print(f"[LLM] ✗ Error: {error_msg}")
        last_llm_analysis = {
            "insight": error_msg,
            "timestamp": time.strftime('%H:%M:%S'),
            "ph": round(ph, 2),
            "tb": round(tb, 2),
            "tds": round(tds, 2),
            "reading_count": reading_count
        }


def analyze_water_quality_groq(ph, turbidity, tds):
    """Analyze water quality using Groq LLM with Sabarmati-specific context."""
    
    # Simplified system prompt
    system_prompt = "You are Dr. Sharma, a water quality expert for Sabarmati River in Ahmedabad, India. Analyze sensor data and provide brief, clear assessments in 2-3 sentences. Use simple English. Focus on water safety and river health."

    # Direct user prompt - simple and clear
    user_prompt = f"""Sabarmati River monitoring station reading:
pH: {ph:.1f}, Turbidity: {turbidity:.1f} NTU, TDS: {tds:.0f} ppm

Provide a 2-3 sentence assessment of water quality and concerns for the river ecosystem."""

    # Call Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,  # Very low for consistent output
        "max_tokens": 100,   # Strict limit for 2-3 sentences
        "top_p": 0.9
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        insight = result['choices'][0]['message']['content'].strip()
        
        # Clean up any encoding issues
        insight = insight.replace('\r', '').replace('\n\n', ' ')
        
        return insight
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return f"Analysis error: {str(e)[:50]}"


@app.route('/sensor-data', methods=['GET'])
def get_sensor_data():
    """Get latest sensor data."""
    distance_value = latest_water_level.get('distance')
    print(f"[API] /sensor-data called, distance = {distance_value}")
    return jsonify({
        'ok': True,
        'data': latest_sensor_data,
        'location': latest_location,
        'distance': distance_value,
        'subscribed': subscription_active
    })


@app.route('/sensor-history', methods=['GET'])
def get_sensor_history():
    """Get sensor data history."""
    return jsonify({
        'ok': True,
        'history': sensor_data_history,
        'locationHistory': location_history,
        'count': len(sensor_data_history)
    })


@app.route('/api/sensor-locations', methods=['GET'])
def get_sensor_locations():
    """Endpoint returning current and historical locations for map rendering."""
    return jsonify({
        'ok': True,
        'current': latest_location,
        'history': location_history,
        'count': len(location_history)
    })


@app.route('/water-level', methods=['GET'])
def get_water_level():
    """Get latest water level data from ultrasonic sensor."""
    return jsonify(latest_water_level)


@app.route('/api/llm-insights', methods=['GET'])
def get_llm_insights():
    """Get latest LLM water quality analysis."""
    return jsonify(last_llm_analysis)


@app.route('/publish', methods=['POST'])
def publish_message():
    """Endpoint for publishing water quality data to MQTT."""
    try:
        data = request.get_json()
        topic = data.get('topic', PUBLISH_TOPIC)
        message = data.get('message', '')
        
        if not message:
            return jsonify({'ok': False, 'error': 'Message cannot be empty'}), 400
        
        client = mqtt.Client()
        client.connect(BROKER, PORT, keepalive=5)
        client.loop_start()
        time.sleep(0.5)
        info = client.publish(topic, message, qos=1)
        print(f"Published to {topic}: {message}")
        time.sleep(0.2)
        client.loop_stop()
        client.disconnect()
        
        return jsonify({'ok': True, 'message': message, 'topic': topic})
    except Exception as e:
        print(f"Publish error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/')
def index():
    """Serve the monitoring dashboard."""
    index_path = os.path.join(BASE_DIR, 'index.html')
    with open(index_path, 'r', encoding='utf-8') as f:
        html = f.read()
    return Response(html, mimetype='text/html')


@app.route('/map_marker.png')
def serve_map_marker():
    """Serve the map marker icon."""
    marker_path = os.path.join(BASE_DIR, 'map_marker.png')
    if os.path.exists(marker_path):
        return send_file(marker_path, mimetype='image/png')
    print(f"[ERROR] Marker file not found at: {marker_path}")
    return jsonify({'error': 'Marker not found'}), 404


@app.route('/DeepFish.gif')
def serve_deepfish_gif():
    """Serve the DeepFish detection GIF."""
    gif_file = 'DeepFish.gif'
    print(f"[GIF] DeepFish.gif requested")
    print(f"[GIF] File path: {os.path.abspath(gif_file)}")
    print(f"[GIF] File exists: {os.path.exists(gif_file)}")
    
    if not os.path.exists(gif_file):
        print(f"[GIF] ERROR: DeepFish.gif not found!")
        return jsonify({'error': 'DeepFish.gif not found', 'path': os.path.abspath(gif_file)}), 404

    return send_file(gif_file, mimetype='image/gif', conditional=True)


@app.route('/OzFish.gif')
def serve_ozfish_gif():
    """Serve the OzFish detection GIF."""
    gif_file = 'OzFish.gif'
    print(f"[GIF] OzFish.gif requested")
    print(f"[GIF] File path: {os.path.abspath(gif_file)}")
    print(f"[GIF] File exists: {os.path.exists(gif_file)}")
    
    if not os.path.exists(gif_file):
        print(f"[GIF] ERROR: OzFish.gif not found!")
        return jsonify({'error': 'OzFish.gif not found', 'path': os.path.abspath(gif_file)}), 404
    
    return send_file(gif_file, mimetype='image/gif', conditional=True)


@app.route('/api/maps-key')
def get_maps_key():
    """Provide Google Maps API key to the frontend."""
    api_key = GOOGLE_MAPS_API_KEY
    print(f"[DEBUG] API key requested: {api_key[:10] if api_key else 'None'}...")
    
    if not api_key or api_key == 'None':
        print("[ERROR] Google Maps API key is missing or None!")
        return jsonify({
            'ok': False,
            'error': 'API key not configured'
        })
    
    response = jsonify({
        'ok': True,
        'apiKey': api_key
    })
    # Add cache control headers to prevent browser caching issues
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/weather')
def get_weather():
    """Fetch weather data for Sabarmati Riverfront using Open-Meteo (free, no API key needed)."""
    import requests
    try:
        # Sabarmati Riverfront coordinates
        lat = 23.0276
        lon = 72.5715
        
        # Using Open-Meteo API - Free, accurate, no API key required!
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=Asia/Kolkata'
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200 and 'current' in data:
            current = data['current']
            
            # Map weather codes to descriptions
            weather_code = current.get('weather_code', 0)
            weather_descriptions = {
                0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
                45: 'foggy', 48: 'depositing rime fog', 51: 'light drizzle', 53: 'moderate drizzle',
                55: 'dense drizzle', 61: 'slight rain', 63: 'moderate rain', 65: 'heavy rain',
                71: 'slight snow', 73: 'moderate snow', 75: 'heavy snow', 80: 'slight rain showers',
                81: 'moderate rain showers', 82: 'violent rain showers', 95: 'thunderstorm',
                96: 'thunderstorm with slight hail', 99: 'thunderstorm with heavy hail'
            }
            description = weather_descriptions.get(weather_code, 'clear')
            
            # Map weather codes to icon categories
            if weather_code == 0 or weather_code == 1:
                icon = 'Clear'
            elif weather_code in [2, 3]:
                icon = 'Clouds'
            elif weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                icon = 'Rain'
            elif weather_code in [71, 73, 75]:
                icon = 'Snow'
            elif weather_code in [95, 96, 99]:
                icon = 'Thunderstorm'
            elif weather_code in [45, 48]:
                icon = 'Fog'
            else:
                icon = 'Clear'
            
            return jsonify({
                'ok': True,
                'temp': round(current['temperature_2m']),
                'description': description,
                'humidity': current['relative_humidity_2m'],
                'wind': round(current['wind_speed_10m']),  # Already in km/h
                'pressure': round(current['surface_pressure']),
                'visibility': 10.0,  # Open-Meteo doesn't provide visibility, default to 10km
                'icon': icon
            })
        else:
            return jsonify({
                'ok': False,
                'error': 'Weather data unavailable'
            }), 400
    except Exception as e:
        print(f"Weather API error: {e}")
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


# Drowning Detection Video Stream
drowning_model = None
drowning_detection_disabled_reason = None  # Set when detection fails so we can fall back to raw video
drowning_video_path = os.path.join(BASE_DIR, "drowning detection", "Automated-Drowning-Detection-YOLOV8", "vdo1.mp4")
model_path = os.path.join(BASE_DIR, "drowning detection", "Automated-Drowning-Detection-YOLOV8", "model.pt")
detection_stats = {
    "status": "SAFE",
    "in_water": 0,
    "risk_level": "low",
    "alert_type": "normal",
    "continuous_duration": 0,
    "risk_count_7min": 0,
    "model_loaded": False,
    "disabled_reason": None
}

# Risk tracking
risk_start_time = None
continuous_risk_duration = 0
risk_events = []  # List of timestamps when risk was detected
RISK_CONTINUOUS_THRESHOLD = 7  # seconds
RISK_FREQUENCY_WINDOW = 420  # 7 minutes in seconds
RISK_FREQUENCY_THRESHOLD = 20  # number of events

# Cache for latest detections to avoid blocking frame display
latest_detections = None
detection_lock = threading.Lock()

def load_drowning_model_async():
    """Preload drowning model in background thread at startup"""
    global drowning_model, drowning_detection_disabled_reason
    
    try:
        print(f"[DROWNING] Loading YOLOv4-tiny with OpenCV DNN in background...")
        
        weights_path = "yolov4-tiny.weights"
        config_path = "yolov4-tiny.cfg"
        
        # Check files exist
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            drowning_detection_disabled_reason = "Model files missing"
            print(f"❌ [DROWNING] Model files not found")
            return
        
        # Load the network
        print(f"[DROWNING] cv2.dnn.readNet starting...")
        net = cv2.dnn.readNet(weights_path, config_path)
        print(f"[DROWNING] cv2.dnn.readNet completed")
        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        drowning_model = {
            "type": "yolo_dnn",
            "net": net,
            "output_layers": output_layers
        }
        
        print(f"✅ [DROWNING] YOLOv4-tiny loaded successfully")
        drowning_detection_disabled_reason = None
    except Exception as e:
        drowning_detection_disabled_reason = f"Model load failed: {e}"
        print(f"❌ [DROWNING] Failed to load model: {e}")

def generate_drowning_detection_live():
    """Generate live camera stream with drowning detection using fast OpenCV DNN"""
    global detection_stats
    
    print(f"[DROWNING] Stream request received")
    
    # Try to open USB camera
    cap = None
    for cam_idx in [1, 2, 0]:
        try:
            test_cap = cv2.VideoCapture(cam_idx)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    print(f"[DROWNING] Using camera {cam_idx}")
                    cap = test_cap
                    break
                test_cap.release()
        except:
            pass
    
    # Fallback to video if no camera
    if cap is None:
        print(f"[DROWNING] Using video file: {drowning_video_path}")
        cap = cv2.VideoCapture(drowning_video_path)
    
    if not cap.isOpened():
        print(f"[DROWNING] ERROR: Could not open camera or video")
        # Return error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera not available", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for faster processing
    
    print(f"[DROWNING] Stream started")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Run detection on every frame if model available
        if drowning_model is not None and drowning_model.get("type") == "yolo_dnn":
            try:
                net = drowning_model["net"]
                output_layers = drowning_model["output_layers"]
                
                # Create blob
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                
                # Forward pass
                outputs = net.forward(output_layers)
                
                # Process detections
                boxes = []
                confidences = []
                class_ids = []
                
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > 0.3:  # Person class (0)
                            if class_id == 0:  # Person
                                x = int(detection[0] * w)
                                y = int(detection[1] * h)
                                width = int(detection[2] * w)
                                height = int(detection[3] * h)
                                
                                x1 = max(0, x - width // 2)
                                y1 = max(0, y - height // 2)
                                x2 = min(w, x + width // 2)
                                y2 = min(h, y + height // 2)
                                
                                boxes.append([x1, y1, x2, y2])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                
                # NMS to remove overlapping boxes
                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
                    
                    for i in indices:
                        i = i[0] if isinstance(i, np.ndarray) else i
                        x1, y1, x2, y2 = boxes[i]
                        conf = confidences[i]
                        
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"PERSON: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception as e:
                print(f"[DROWNING] Detection error: {e}")
        else:
            # Show status while model loads
            if drowning_detection_disabled_reason:
                cv2.putText(frame, "Model error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Loading model...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Encode and stream
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1.0 / 15.0)  # 15 FPS for smooth playback  # 15 FPS for smooth playback

@app.route('/video/drowning')
def drowning_video_feed():
    """Serve drowning detection live stream"""
    try:
        print(f"[VIDEO] Live stream requested")
        return Response(generate_drowning_detection_live(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"[VIDEO ERROR] {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/video/drowning-file')
def drowning_video_file():
    """Serve the Drowning_video.mp4 file with range support for video playback"""
    video_file = 'Drowning_video.mp4'
    
    print(f"[VIDEO] Drowning video file requested")
    print(f"[VIDEO] File path: {os.path.abspath(video_file)}")
    print(f"[VIDEO] File exists: {os.path.exists(video_file)}")
    
    if not os.path.exists(video_file):
        print(f"[VIDEO] ERROR: Video file not found!")
        return jsonify({'error': 'Video file not found', 'path': os.path.abspath(video_file)}), 404
    
    # Get file size
    file_size = os.path.getsize(video_file)
    print(f"[VIDEO] File size: {file_size / (1024*1024):.2f} MB")
    
    # Parse Range header
    range_header = request.headers.get('Range', None)
    print(f"[VIDEO] Range header: {range_header}")
    
    if not range_header:
        # No range requested, send entire file
        print(f"[VIDEO] Sending full file")
        return send_file(video_file, mimetype='video/mp4', conditional=True)
    
    # Parse range request
    byte_range = range_header.replace('bytes=', '').split('-')
    start = int(byte_range[0]) if byte_range[0] else 0
    end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
    
    print(f"[VIDEO] Serving range: {start}-{end} of {file_size}")
    
    # Read the requested chunk
    chunk_size = end - start + 1
    
    def generate():
        with open(video_file, 'rb') as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                read_size = min(8192, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data
    
    # Build response with partial content
    response = Response(generate(), 206, mimetype='video/mp4', direct_passthrough=True)
    response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Content-Length', str(chunk_size))
    response.headers.add('Cache-Control', 'no-cache')
    
    return response

@app.route('/detection/test')
def detection_test():
    """Test if model and video are accessible"""
    import os
    video_exists = os.path.exists(drowning_video_path)
    model_exists = os.path.exists(model_path)
    return jsonify({
        'video_path': drowning_video_path,
        'video_exists': video_exists,
        'model_path': model_path,
        'model_exists': model_exists,
        'model_loaded': drowning_model is not None,
        'disabled_reason': drowning_detection_disabled_reason,
        'stats': detection_stats
    })

@app.route('/detection/status')
def drowning_status():
    """Get current drowning detection status"""
    return jsonify(detection_stats)


# ============================================
# BIODIVERSITY CAMERA - LIVE USB DETECTION
# ============================================

biodiversity_model = None
bio_camera_active = False
bio_detection_stats = {
    "objects_detected": 0,
    "species": [],
    "confidence_avg": 0.0
}

def create_error_frame(message):
    """Create an error frame image with a message"""
    import numpy as np
    # Create black frame with text
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera Error", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, message, (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Check terminal for details", (130, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def load_biodiversity_model_opencv():
    """Load YOLO model using OpenCV DNN (no numpy issues)"""
    global biodiversity_model
    if biodiversity_model is None:
        try:
            import os
            import urllib.request
            
            # COCO class names (80 classes)
            class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                          "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                          "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            
            # Try to load YOLOv4-tiny with OpenCV DNN
            weights_path = "yolov4-tiny.weights"
            config_path = "yolov4-tiny.cfg"
            
            # Download files if not exists
            if not os.path.exists(config_path):
                print("[BioCam] Downloading YOLOv4-tiny config...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
                    config_path
                )
            
            if not os.path.exists(weights_path):
                print("[BioCam] Downloading YOLOv4-tiny weights (23 MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
                    weights_path
                )
            
            # Load the network
            net = cv2.dnn.readNet(weights_path, config_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            biodiversity_model = {
                "type": "yolo_dnn",
                "net": net,
                "output_layers": output_layers,
                "classes": class_names
            }
            
            print(f"✅ YOLOv4-tiny loaded successfully with {len(class_names)} classes")
            return biodiversity_model
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print(f"⚠️ Falling back to basic detection")
            
            # Fallback to basic detection
            class_names = ["person", "bottle", "cup", "potted plant", "bird", "cat", "dog"]
            biodiversity_model = {
                "type": "basic",
                "classes": class_names
            }
            return biodiversity_model
    return biodiversity_model

def generate_biodiversity_camera():
    """Generate live USB camera stream with object detection"""
    global bio_camera_active, bio_detection_stats
    
    print(f"[BioCam] Starting live camera detection...")
    
    # Load model (will use OpenCV-based detection)
    model = load_biodiversity_model_opencv()
    # model = load_biodiversity_model()
    # if model is None:
    #     print("❌ Model not loaded, cannot stream")
    #     error_frame = create_error_frame("Model not loaded")
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
    #     return
    
    # Try to open USB camera - try index 1 first (usually external USB camera), then 2, then 0 (built-in)
    cap = None
    camera_indices = [1, 2, 3, 0]  # Try USB cameras first, built-in last
    
    for cam_index in camera_indices:
        print(f"[BioCam] Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Test if we can actually read from it
            ret, test_frame = cap.read()
            if ret:
                print(f"✅ [BioCam] Camera {cam_index} opened successfully!")
                break
            else:
                print(f"❌ [BioCam] Camera {cam_index} opened but can't read frames")
                cap.release()
                cap = None
        else:
            print(f"❌ [BioCam] Camera {cam_index} not available")
    
    if cap is None or not cap.isOpened():
        print(f"❌ [BioCam] No camera found after trying all indices")
        error_frame = create_error_frame("No camera detected")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Store which camera index was used
    used_camera_index = camera_indices[camera_indices.index(cam_index)] if cap else 0
    
    print(f"[BioCam] Using camera index {used_camera_index}, streaming...")
    bio_camera_active = True
    
    frame_count = 0
    detected_objects_count = 0
    
    # Cache for detection results to avoid re-processing every frame
    last_boxes = []
    last_confidences = []
    last_class_ids = []
    last_hog_rects = []  # Cache for HOG detections
    last_hog_weights = []
    detection_frame_skip = 3  # Process every 3rd frame for detection
    
    # Initialize HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Get model info
    model_type = model.get("type", "basic") if model else "basic"
    class_names = model.get("classes", []) if model else []
    
    # Classes we care about for biodiversity monitoring
    target_classes = ["person", "bird", "cat", "dog", "horse", "cow", "bottle", "potted plant", "vase"]
    
    while bio_camera_active:
        ret, frame = cap.read()
        
        if not ret:
            print("[BioCam] Failed to read frame")
            break
        
        frame_count += 1
        detected_objects_count = 0
        detected_objects = []
        
        # Use YOLO DNN if available
        if model_type == "yolo_dnn":
            height, width = frame.shape[:2]
            
            # Only run detection every Nth frame to maintain smooth video
            if frame_count % detection_frame_skip == 0:
                # Create blob from image
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                model["net"].setInput(blob)
                
                # Forward pass
                outputs = model["net"].forward(model["output_layers"])
                
                # Process detections
                boxes = []
                confidences = []
                class_ids = []
                
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = int(scores.argmax())
                        confidence = float(scores[class_id])
                        
                        # Filter by confidence and target classes
                        if confidence > 0.3 and class_names[class_id] in target_classes:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(confidence)
                            class_ids.append(class_id)
                
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
                
                last_boxes = boxes
                last_confidences = confidences
                last_class_ids = class_ids
            
            if len(last_boxes) > 0:
                indices = cv2.dnn.NMSBoxes(last_boxes, last_confidences, 0.3, 0.4)
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = last_boxes[i]
                        label = class_names[last_class_ids[i]]
                        conf = last_confidences[i]
                        
                        if label == "person":
                            color = (0, 255, 0) 
                        elif label in ["bird", "cat", "dog", "horse", "cow"]:
                            color = (255, 100, 0) 
                        elif label == "potted plant" or label == "vase":
                            color = (0, 200, 100) 
                        else:
                            color = (0, 165, 255) 
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame, (x, y - 25), (x + 100, y), color, -1)
                        display_label = "human" if label == "person" else label
                        cv2.putText(frame, display_label, (x + 5, y - 7), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detected_objects.append(label)
                        detected_objects_count += 1
        
        else:
            # Fallback: Use HOG for people detection only
            if frame_count % detection_frame_skip == 0:
                scale = 0.5
                small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Detect people
                (rects, weights) = hog.detectMultiScale(small_frame, winStride=(4, 4),
                                                       padding=(8, 8), scale=1.05)
                
                # Cache HOG results
                last_hog_rects = rects
                last_hog_weights = weights
            
            # Draw cached HOG detections
            if len(last_hog_rects) > 0:
                scale = 0.5
                # Draw rectangles around detected people
                for i, (x, y, w, h) in enumerate(last_hog_rects):
                    # Scale back to original size
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y - 25), (x + 100, y), (0, 200, 0), -1)
                    cv2.putText(frame, "human", (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    detected_objects.append("person")
                    detected_objects_count += 1        # Update stats
        bio_detection_stats = {
            "objects_detected": detected_objects_count,
            "species": list(set(detected_objects)),  # Unique objects
            "confidence_avg": 0.75 if detected_objects_count > 0 else 0.0
        }
        
        # Add text overlays
        camera_type = "USB CAMERA" if used_camera_index > 0 else "BUILT-IN CAMERA"
        cv2.putText(frame, f"LIVE {camera_type} (Index: {used_camera_index})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {detected_objects_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if detected_objects_count > 0:
            cv2.putText(frame, f"Types: {', '.join(set(detected_objects))}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add overlay info
        info_text = f"Camera Active | Frame: {frame_count}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Cleanup
    cap.release()
    bio_camera_active = False
    print("[BioCam] Camera stream stopped")

@app.route('/test_camera')
def test_camera():
    """Test page for camera debugging"""
    return send_file('test_camera.html')

@app.route('/video/test_raw')
def test_raw_camera():
    """Simple raw camera feed without detection for testing"""
    def generate():
        print("[TestCam] Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("[TestCam] Failed to open camera 0, trying 1...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("[TestCam] No camera found")
            # Create error frame
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Found", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        print("[TestCam] Camera opened successfully")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[TestCam] Failed to read frame")
                break
            
            # Add text overlay
            cv2.putText(frame, "RAW CAMERA FEED - TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
        print("[TestCam] Camera released")
    
    print("[TestCam] Stream endpoint called")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/biodiversity')
def biodiversity_video_feed():
    """Biodiversity live camera detection stream endpoint"""
    print(f"[BioCam] ===== Stream endpoint called =====")
    try:
        print(f"[BioCam] Calling generate_biodiversity_camera()")
        return Response(generate_biodiversity_camera(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"[BioCam ERROR] Exception in endpoint: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/biodiversity/status')
def biodiversity_status():
    """Get current biodiversity detection status"""
    return jsonify(bio_detection_stats)

@app.route('/biodiversity/stop', methods=['POST'])
def stop_biodiversity_camera():
    """Stop the biodiversity camera stream"""
    global bio_camera_active
    bio_camera_active = False
    return jsonify({'ok': True, 'message': 'Camera stopped'})


if __name__ == '__main__':
    print(f"Starting Water Quality Monitoring Server...")
    print(f"MQTT Broker: {BROKER}:{PORT}")
    print(f"Subscribe Topic: {SUBSCRIBE_TOPIC}")
    print(f"Publish Topic: {PUBLISH_TOPIC}")
    
    # Preload drowning model in background thread
    print(f"[DROWNING] Preloading model...")
    model_thread = threading.Thread(target=load_drowning_model_async, daemon=True)
    model_thread.start()
    
    # Auto-start MQTT subscription
    print(f"[MQTT] Auto-starting subscription to {SUBSCRIBE_TOPIC}...")
    start_mqtt_subscription()
    
    print(f"Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
