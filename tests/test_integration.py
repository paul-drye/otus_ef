#!/usr/bin/env python3
import os
import sys
import argparse
import time

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision_engine import DecisionEngine
from src.face_recognition_module import FaceRecognizer
from src.voice_recognition_module import VoiceRecognizer


def test_full_enrollment_verification_flow(user_name="Test_Integration_User"):
    """
    Test the full enrollment and verification flow with both face and voice.
    
    Args:
        user_name (str): Name to use for the test user
    """
    print(f"\n=== Testing Full Enrollment and Verification Flow for '{user_name}' ===")
    
    # Initialize decision engine
    engine = DecisionEngine()
    
    # 1. ENROLLMENT PHASE
    print("\n1. ENROLLMENT PHASE")
    print("-" * 50)
    
    # Enroll the user
    print(f"Enrolling user '{user_name}' with face and voice...")
    enrollment_success = engine.enroll_user(user_name, face_samples=3, voice_samples=1)
    
    if not enrollment_success:
        print("❌ Enrollment failed. Aborting test.")
        return False
    
    print(f"✅ Successfully enrolled user '{user_name}' with face and voice.")
    
    # Verify the user exists in both systems
    face_users = engine.face_recognizer.known_face_names
    voice_users = list(engine.voice_recognizer.speaker_embeddings.keys())
    
    if user_name in face_users:
        print(f"✅ User '{user_name}' found in face recognition database.")
    else:
        print(f"❌ User '{user_name}' NOT found in face recognition database.")
        return False
        
    if user_name in voice_users:
        embedding = engine.voice_recognizer.speaker_embeddings[user_name]
        print(f"✅ User '{user_name}' found in voice recognition database.")
        print(f"   Voice embedding shape: {embedding.shape}")
    else:
        print(f"❌ User '{user_name}' NOT found in voice recognition database.")
        return False
    
    # 2. VERIFICATION PHASE
    print("\n2. VERIFICATION PHASE")
    print("-" * 50)
    
    print("Running full authentication process...")
    print("Please position yourself in front of the camera and then speak when prompted.")
    
    authenticated, verified_name = engine.authenticate_user(face_time=10, display_video=True)
    
    if authenticated and verified_name == user_name:
        print(f"✅ User '{user_name}' successfully authenticated with both face and voice!")
        return True
    else:
        if not authenticated:
            print("❌ Authentication failed.")
        else:
            print(f"❌ Authentication returned wrong user: '{verified_name}' instead of '{user_name}'")
        return False


def test_face_only_verification(user_name=None):
    """
    Test face recognition verification in isolation.
    
    Args:
        user_name (str, optional): Expected user name
    """
    print("\n=== Testing Face Recognition Verification ===")
    
    face_recognizer = FaceRecognizer()
    
    if not face_recognizer.known_face_names:
        print("No enrolled face users found. Please enroll a user first.")
        return False
    
    if user_name is None:
        print(f"Available users: {', '.join(face_recognizer.known_face_names)}")
        print("Expected user name not provided. Will accept any known face.")
    
    print("Attempting face recognition...")
    print("Please position yourself in front of the camera.")
    
    recognized, name = face_recognizer.recognize_face(display_video=True, recognition_time=10)
    
    if recognized:
        if user_name is None or name == user_name:
            print(f"✅ Face successfully recognized as '{name}'!")
            return True
        else:
            print(f"❌ Face recognized as '{name}' instead of '{user_name}'")
            return False
    else:
        print("❌ Face not recognized.")
        return False


def test_voice_only_verification(user_name=None):
    """
    Test voice verification in isolation.
    
    Args:
        user_name (str, optional): Expected user name
    """
    print("\n=== Testing Voice Verification ===")
    
    voice_recognizer = VoiceRecognizer()
    
    if not voice_recognizer.speaker_embeddings:
        print("No enrolled voice users found. Please enroll a user first.")
        return False
    
    if user_name is None:
        print(f"Available users: {', '.join(voice_recognizer.speaker_embeddings.keys())}")
        print("Expected user name not provided. Will accept any known voice.")
    
    print("Attempting voice verification...")
    print("Please speak when ready.")
    
    verified, name = voice_recognizer.verify_speaker()
    
    if verified:
        if user_name is None or name == user_name:
            print(f"✅ Voice successfully verified as '{name}'!")
            return True
        else:
            print(f"❌ Voice verified as '{name}' instead of '{user_name}'")
            return False
    else:
        print("❌ Voice not verified.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Face and Voice Integration Tests')
    
    subparsers = parser.add_subparsers(dest='command', help='Test command')
    
    # Full flow test
    full_flow_parser = subparsers.add_parser('full-flow', help='Test full enrollment and verification flow')
    full_flow_parser.add_argument('--name', type=str, default="Test_Integration_User", help='Name for the test user')
    
    # Face verification test
    face_parser = subparsers.add_parser('face', help='Test face recognition verification')
    face_parser.add_argument('--name', type=str, help='Expected user name (optional)')
    
    # Voice verification test
    voice_parser = subparsers.add_parser('voice', help='Test voice verification')
    voice_parser.add_argument('--name', type=str, help='Expected user name (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'full-flow':
        test_full_enrollment_verification_flow(args.name)
    elif args.command == 'face':
        test_face_only_verification(args.name)
    elif args.command == 'voice':
        test_voice_only_verification(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 