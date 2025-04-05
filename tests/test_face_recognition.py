#!/usr/bin/env python3
import cv2
import argparse
import time
import os
import sys

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.face_recognition_module import FaceRecognizer


def test_face_detection(display_time=10):
    """
    Test if face detection is working properly.
    
    Args:
        display_time (int): How long to display the video feed in seconds
    
    Returns:
        bool: True if face detection works, False otherwise
    """
    print("Testing face detection...")
    
    # Create face recognizer
    recognizer = FaceRecognizer()
    
    # Test the camera with face detection
    return recognizer.test_camera(display_time=display_time)


def test_face_enrollment(user_name="Test User", samples=3):
    """
    Test face enrollment functionality.
    
    Args:
        user_name (str): Name for the test user
        samples (int): Number of face samples to capture
    
    Returns:
        bool: True if enrollment succeeds, False otherwise
    """
    print(f"Testing face enrollment for user '{user_name}'...")
    
    # Create face recognizer
    recognizer = FaceRecognizer()
    
    # Enroll user
    return recognizer.enroll_user(user_name, num_samples=samples)


def test_face_recognition(display_time=10):
    """
    Test face recognition functionality.
    
    Args:
        display_time (int): How long to run recognition
    
    Returns:
        tuple: (recognized, user_name) - whether a face was recognized and the user's name
    """
    print("Testing face recognition...")
    
    # Create face recognizer
    recognizer = FaceRecognizer()
    
    # Check if there are enrolled users
    if not recognizer.known_face_names:
        print("No enrolled users found. Please enroll a user first.")
        return False, None
    
    # Run recognition
    return recognizer.recognize_face(display_video=True, recognition_time=display_time)


def main():
    parser = argparse.ArgumentParser(description='Test Face Recognition Module')
    
    # Create subparsers for different test types
    subparsers = parser.add_subparsers(dest='test_type', help='Type of test to run')
    
    # Detection test
    detection_parser = subparsers.add_parser('detection', help='Test face detection')
    detection_parser.add_argument('--time', type=int, default=10, 
                                help='Time in seconds to run the test')
    
    # Enrollment test
    enrollment_parser = subparsers.add_parser('enrollment', help='Test user enrollment')
    enrollment_parser.add_argument('--name', type=str, default="Test User", 
                                 help='Name for the test user')
    enrollment_parser.add_argument('--samples', type=int, default=3, 
                                 help='Number of face samples to capture')
    
    # Recognition test
    recognition_parser = subparsers.add_parser('recognition', help='Test face recognition')
    recognition_parser.add_argument('--time', type=int, default=10, 
                                  help='Time in seconds to run recognition')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.test_type == 'detection':
        success = test_face_detection(display_time=args.time)
        if success:
            print("Face detection test passed!")
        else:
            print("Face detection test failed.")
            
    elif args.test_type == 'enrollment':
        success = test_face_enrollment(user_name=args.name, samples=args.samples)
        if success:
            print(f"Successfully enrolled user '{args.name}'")
        else:
            print(f"Failed to enroll user '{args.name}'")
            
    elif args.test_type == 'recognition':
        recognized, name = test_face_recognition(display_time=args.time)
        if recognized:
            print(f"Recognition successful! Recognized user: {name}")
        else:
            print("Recognition failed - no matching face found.")
            
    else:
        # Default: run all tests in sequence
        print("No specific test type selected. Please choose one of the following:")
        print("  python -m tests.test_face_recognition detection")
        print("  python -m tests.test_face_recognition enrollment --name \"Your Name\"")
        print("  python -m tests.test_face_recognition recognition")
        parser.print_help()


if __name__ == "__main__":
    main() 