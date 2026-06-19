"""
Drowning Detection - Main Runner Script
Easy to use interface for training and detection
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def print_menu():
    print("\n" + "="*60)
    print("🏊 DROWNING DETECTION SYSTEM")
    print("="*60)
    print("\nSelect an option:")
    print("  1. Quick Train (10 epochs - for testing)")
    print("  2. Full Train (100 epochs - for production)")
    print("  3. Detect from Webcam")
    print("  4. Detect from Video File")
    print("  5. Validate Model")
    print("  6. Exit")
    print("-"*60)

def main():
    while True:
        print_menu()
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            print("\n⚡ Starting Quick Training...")
            from train_model import train_drowning_model
            train_drowning_model(epochs=10, batch_size=8, model_size='n')
            
        elif choice == '2':
            print("\n🎯 Starting Full Training...")
            from train_model import train_drowning_model
            train_drowning_model(epochs=100, batch_size=16, model_size='s')
            
        elif choice == '3':
            print("\n📹 Starting Webcam Detection...")
            from detect_drowning import detect_video
            detect_video(0)
            
        elif choice == '4':
            video_path = input("\nEnter video file path: ").strip()
            if video_path:
                print(f"\n🎬 Processing video: {video_path}")
                from detect_drowning import detect_video
                output_path = SCRIPT_DIR / "output_detected.mp4"
                detect_video(video_path, output_path=str(output_path))
            else:
                print("❌ No video path provided")
                
        elif choice == '5':
            print("\n📊 Validating Model...")
            from train_model import validate_model
            validate_model()
            
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()
