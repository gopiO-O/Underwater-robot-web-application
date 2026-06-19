#!/usr/bin/env python3
"""
Sensor Data Publisher - Simulates sensor data and publishes to MQTT.

Sends random pH, Turbidity, and TDS values in JSON format:
{"ph": 7.2, "tb": 15.5, "tds": 320}

Run from any computer that can reach the MQTT broker.

Example:
  python sensor_publisher.py --broker 44.214.52.220 --count 10 --interval 1
"""
import argparse
import time
import random
import json
import sys
import paho.mqtt.client as mqtt


def publish_sensor_data(broker, port, topic, interval):
    """Publish simulated sensor data to MQTT broker continuously."""
    client = mqtt.Client()
    
    print(f"Connecting to {broker}:{port}...")
    client.connect(broker, port, 60)
    print(f"Connected to MQTT broker")
    print(f"Publishing continuously... (Press Ctrl+C to stop)\n")
    
    # Sabarmati Riverfront GPS coordinates (slight variation for movement simulation)
    base_lat = 23.0276
    base_lon = 72.5715
    
    message_count = 0
    
    try:
        while True:
            message_count += 1
            
            # Generate random sensor values
            sensor_data = {
                "ph": round(random.uniform(6.0, 9.0), 1),
                "tb": round(random.uniform(0.5, 50.0), 1),
                "tds": random.randint(100, 500),
                "lat": round(base_lat + random.uniform(-0.005, 0.005), 6),
                "lon": round(base_lon + random.uniform(-0.005, 0.005), 6)
            }
            
            # Convert to JSON string
            message = json.dumps(sensor_data)
            
            # Publish to MQTT
            client.publish(topic, message)
            print(f"[{message_count}] Sent: {message}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n\nStopped. Total messages sent: {message_count}")
    
    finally:
        client.disconnect()
        print("Disconnected from broker")


def parse_args():
    p = argparse.ArgumentParser(description='Sensor Data MQTT Publisher (Continuous)')
    p.add_argument('--broker', default='44.214.52.220', help='MQTT broker host or IP')
    p.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    p.add_argument('--topic', default='SIH/Gopi/pub', help='Topic to publish sensor data')
    p.add_argument('--interval', type=float, default=2.0, help='Seconds between messages')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Sensor Data Publisher (Continuous)")
    print(f"=====================================")
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topic: {args.topic}")
    print(f"Interval: {args.interval}s")
    print(f"Format: {{\"ph\": X.X, \"tb\": X.X, \"tds\": XXX, \"lat\": XX.XXXX, \"lon\": XX.XXXX}}")
    print(f"Location: Sabarmati Riverfront, Ahmedabad")
    print(f"=====================================")
    print()
    
    publish_sensor_data(args.broker, args.port, args.topic, args.interval)


if __name__ == '__main__':
    main()
