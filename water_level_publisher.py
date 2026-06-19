import paho.mqtt.client as mqtt
import json
import time
import random

# MQTT Configuration
BROKER = "44.214.52.220"
PORT = 1883
TOPIC = "SIH/Gopi/pub"

# Water Level Configuration
# Sensor is fixed above water at a certain height
# Normal safe distance: 100 cm
# As water rises (flood), distance decreases
NORMAL_DISTANCE = 100  # cm (safe level)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker at {BROKER}:{PORT}")
    else:
        print(f"Connection failed with code {rc}")

def on_publish(client, userdata, mid):
    print(f"Message {mid} published")

def publish_water_level(client):
    """
    Publishes ultrasonic sensor data.
    The sensor is fixed at a position and measures distance to water surface.
    Normal safe level: 100 cm (sensor to water)
    As water rises, distance decreases (flood risk increases)
    
    Range: 0-100 cm
    - 100 cm: Safe (normal level)
    - 50-100 cm: Medium Risk (water rising)
    - 20-50 cm: Risk (water high)
    - 0-20 cm: High Risk (critical flood danger)
    """
    
    # Generate random distance between 0-100 cm
    # Most of the time it should be in safe range (70-100)
    # Occasionally go into risk zones for testing
    
    rand = random.random()
    if rand < 0.6:  # 60% safe
        ultra = random.uniform(70, 100)
    elif rand < 0.8:  # 20% medium risk
        ultra = random.uniform(50, 70)
    elif rand < 0.95:  # 15% risk
        ultra = random.uniform(20, 50)
    else:  # 5% high risk
        ultra = random.uniform(0, 20)
    
    # Send distance in cm
    data = {
        "ultra": round(ultra, 1)
    }
    
    message = json.dumps(data)
    result = client.publish(TOPIC, message, qos=1)
    
    # Determine status for logging
    if ultra >= 100:
        status = "SAFE"
    elif ultra >= 50:
        status = "MEDIUM RISK"
    elif ultra >= 20:
        status = "RISK"
    else:
        status = "HIGH RISK"
    
    print(f"Published: Ultra={ultra:.1f}cm - {status}")
    
    return result

def main():
    print("=== Water Level MQTT Publisher ===")
    print(f"Broker: {BROKER}:{PORT}")
    print(f"Topic: {TOPIC}")
    print(f"Normal Safe Distance: {NORMAL_DISTANCE} cm")
    print("Logic: Distance decreases as water rises (flood risk)")
    print("=" * 40)
    
    # Create MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish
    
    try:
        # Connect to broker
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()
        
        print("Publishing water level data every 2 seconds...")
        print("Press Ctrl+C to stop\n")
        
        while True:
            publish_water_level(client)
            time.sleep(2)  # Publish every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nStopping publisher...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected from broker")

if __name__ == "__main__":
    main()
