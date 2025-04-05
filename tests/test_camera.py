#!/usr/bin/env python3
import cv2
import argparse
import time
import os
import sys

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_camera(camera_index=0, display_time=10):
    """
    Test if the camera is working.
    
    Args:
        camera_index (int): Camera device index
        display_time (int): How long to display the video feed in seconds
    
    Returns:
        bool: True if camera works, False otherwise
    """
    print(f"Testing camera at index {camera_index}...")
    
    # Try to open the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return False
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties: {width}x{height} at {fps} FPS")
    
    # Display camera feed
    print(f"Showing camera feed for {display_time} seconds. Press 'q' to quit earlier.")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < display_time:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Couldn't read frame.")
            break
        
        frame_count += 1
        
        # Display the frame
        cv2.putText(frame, f"Camera {camera_index}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate actual FPS
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"Actual FPS: {actual_fps:.2f}")
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    return True

def scan_cameras(max_index=10):
    """
    Scan for available cameras.
    
    Args:
        max_index (int): Maximum camera index to check
    """
    print(f"Scanning for available cameras (0-{max_index})...")
    
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: Available")
            else:
                print(f"Camera {i}: Available but cannot read frames")
            cap.release()
        else:
            print(f"Camera {i}: Not available")

def update_camera_config(camera_index):
    """
    Update the camera index in the config file.
    
    Args:
        camera_index (int): New camera index to use
    """
    import yaml
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/settings.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update camera index
        config['face_recognition']['camera_index'] = camera_index
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Updated camera index to {camera_index} in {config_path}")
        
    except Exception as e:
        print(f"Error updating config file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test camera for face recognition')
    parser.add_argument('--scan', action='store_true', 
                        help='Scan for available cameras')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera index to test')
    parser.add_argument('--time', type=int, default=10, 
                        help='Time in seconds to display camera feed')
    parser.add_argument('--update-config', action='store_true', 
                        help='Update camera index in config file')
    
    args = parser.parse_args()
    
    if args.scan:
        scan_cameras()
    else:
        if test_camera(args.camera, args.time) and args.update_config:
            update_camera_config(args.camera)

if __name__ == "__main__":
    main() 