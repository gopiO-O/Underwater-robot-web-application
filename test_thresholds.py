#!/usr/bin/env python3
"""
Simple Python script to test threshold values by sending messages to MQTT server
Run this to test if messages are being sent correctly
"""

import paho.mqtt.client as mqtt
import json
import time

# Configuration
BROKER = "44.214.52.220"
PORT = 1883
PUBLISH_TOPIC = "SIH/Vishal/pub"  # Topic to publish threshold data
SUBSCRIBE_TOPIC = "SIH/Gopi/pub"  # Topic to receive sensor data (optional)

# Test threshold data
TEST_THRESHOLDS = {
    "safe_level": 57,          # Safe water level in cm
    "medium_level": 45,        # Medium risk threshold
    "risk_level": 30,          # Risk threshold
    "high_risk_level": 15,     # High risk threshold
    "flood_alert_level": 10,   # Flood alert threshold
}

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        print("✅ Connected to MQTT Broker!")
        print(f"   Broker: {BROKER}:{PORT}")
    else:
        print(f"❌ Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects from the broker"""
    if rc != 0:
        print(f"⚠️  Unexpected disconnection: {rc}")
    else:
        print("✅ Disconnected from MQTT Broker")

def on_publish(client, userdata, mid):
    """Callback when message is published"""
    print(f"✅ Message published successfully (ID: {mid})")

def on_subscribe(client, userdata, mid, granted_qos):
    """Callback when subscription is confirmed"""
    print(f"✅ Subscribed with QoS {granted_qos}")

def on_message(client, userdata, msg):
    """Callback when a message is received"""
    payload = msg.payload.decode('utf-8', errors='ignore')
    print(f"\n📨 Received message on {msg.topic}:")
    try:
        data = json.loads(payload)
        print(f"   {json.dumps(data, indent=2)}")
    except:
        print(f"   {payload}")

def test_send_thresholds():
    """Send test threshold values to the server"""
    print("\n" + "="*60)
    print("MQTT THRESHOLD TEST")
    print("="*60)
    
    try:
        # Create MQTT client
        client = mqtt.Client()
        
        # Set callbacks
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_publish = on_publish
        client.on_subscribe = on_subscribe
        client.on_message = on_message
        
        # Connect to broker
        print(f"\n🔌 Connecting to {BROKER}:{PORT}...")
        client.connect(BROKER, PORT, keepalive=60)
        
        # Start network loop in background
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        # Send 4 identical threshold messages
        threshold_message = json.dumps({"thres":1}, separators=(',', ':'))
        
        for i in range(1, 5):
            print(f"\n📤 Message {i}: Sending threshold...")
            result = client.publish(PUBLISH_TOPIC, threshold_message, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"✅ SUCCESS: Message {i} sent!")
                print(f"   Payload: {threshold_message}")
            else:
                print(f"❌ FAILED to send message {i}: {result.rc}")
            
            time.sleep(1)
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        client.loop_stop()
        client.disconnect()
        
        print("\n" + "="*60)
        print("✅ TEST COMPLETE!")
        print("="*60)
        print("\nSummary:")
        print(f"  • Broker: {BROKER}:{PORT}")
        print(f"  • Publish Topic: {PUBLISH_TOPIC}")
        print(f"  • Subscribe Topic: {SUBSCRIBE_TOPIC}")
        print(f"  • Threshold Values Sent: {json.dumps(TEST_THRESHOLDS, indent=4)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure MQTT broker is running")
        print("  2. Check firewall/network connectivity")
        print("  3. Verify broker IP and port are correct")
        print(f"  4. Check topics are correct:")
        print(f"     - PUBLISH_TOPIC: {PUBLISH_TOPIC}")
        print(f"     - SUBSCRIBE_TOPIC: {SUBSCRIBE_TOPIC}")

if __name__ == "__main__":
    test_send_thresholds()
