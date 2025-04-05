#!/usr/bin/env python3
import argparse
import time
import os
import sys
import traceback

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.voice_recognition_module import VoiceRecognizer


def test_voice_recording(duration=5, output_path="voice_test_recording.wav"):
    """
    Test the voice recording functionality of the VoiceRecognizer.
    
    Args:
        duration (int): Duration in seconds to record
        output_path (str): Path to save the recorded audio
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Testing voice recording...")
    
    # Create voice recognizer
    recognizer = VoiceRecognizer()
    
    # Record audio for the specified duration
    try:
        print(f"Recording audio for {duration} seconds...")
        audio_path = recognizer.record_audio(output_path=output_path, duration=duration)
        print(f"Recording saved to {audio_path}")
        return True, audio_path
    except Exception as e:
        print(f"Error during recording: {e}")
        return False, None


def test_speech_transcription(audio_path=None):
    """
    Test speech-to-text conversion.
    
    Args:
        audio_path (str, optional): Path to an existing audio file to transcribe,
                                   or None to record a new sample
    
    Returns:
        tuple: (success, text) - whether transcription was successful and the transcribed text
    """
    print("Testing speech transcription...")
    
    # Create voice recognizer
    recognizer = VoiceRecognizer()
    
    # If no audio file is provided, record a new one
    if audio_path is None or not os.path.exists(audio_path):
        print("No audio file provided. Recording a new sample...")
        print("Please speak a simple phrase for transcription.")
        _, audio_path = test_voice_recording(duration=5, output_path="transcription_test.wav")
        if not audio_path:
            return False, None
    
    # Transcribe the audio
    try:
        print(f"Transcribing audio from {audio_path}...")
        text = recognizer.transcribe_speech(audio_path=audio_path)
        
        if text:
            print(f"Transcription successful: \"{text}\"")
            return True, text
        else:
            print("Transcription failed. No text was recognized.")
            return False, None
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False, None


def test_speaker_verification_model():
    """
    Test if the speaker verification model is loaded correctly.
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    print("Testing speaker verification model...")
    
    # Create voice recognizer
    recognizer = VoiceRecognizer()
    
    # Check if model is loaded
    if recognizer.model_loaded:
        print("Speaker verification model loaded successfully.")
        return True
    else:
        print("Speaker verification model failed to load.")
        print("Check the error message above for details.")
        return False


def test_user_enrollment(user_name="Test User", num_samples=2):
    """
    Test user enrollment for speaker verification.
    
    Args:
        user_name (str): Name for the test user
        num_samples (int): Number of voice samples to record
        
    Returns:
        bool: True if enrollment succeeds, False otherwise
    """
    print(f"Testing user enrollment for '{user_name}'...")
    
    # Create voice recognizer
    recognizer = VoiceRecognizer()
    
    # Check if model is loaded
    if not recognizer.model_loaded:
        print("Speaker verification model not loaded. Cannot proceed with enrollment test.")
        return False
    
    # Enroll user
    try:
        success = recognizer.enroll_user(user_name, num_samples=num_samples)
        
        if success:
            print(f"Successfully enrolled user '{user_name}'")
        else:
            print(f"Failed to enroll user '{user_name}'")
        
        return success
    
    except Exception as e:
        print(f"Error during user enrollment: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_speaker_verification():
    """
    Test speaker verification functionality.
    
    Returns:
        tuple: (verified, user_name) - whether a speaker was verified and the user's name
    """
    print("Testing speaker verification...")
    
    # Create voice recognizer
    recognizer = VoiceRecognizer()
    
    # Check if model is loaded
    if not recognizer.model_loaded:
        print("Speaker verification model not loaded. Cannot proceed with verification test.")
        return False, None
    
    # Check if there are enrolled users
    if not recognizer.speaker_embeddings:
        print("No enrolled users found. Please enroll a user first.")
        return False, None
    
    # Verify speaker
    try:
        print("Please speak for speaker verification...")
        verified, name = recognizer.verify_speaker()
        
        if verified:
            print(f"Speaker verified as '{name}'")
        else:
            print("Speaker verification failed - no match found.")
        
        return verified, name
    
    except Exception as e:
        print(f"Error during speaker verification: {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description='Test Voice Recognition Module')
    
    # Create subparsers for different test types
    subparsers = parser.add_subparsers(dest='test_type', help='Type of test to run')
    
    # Recording test
    recording_parser = subparsers.add_parser('record', help='Test voice recording')
    recording_parser.add_argument('--duration', type=int, default=5, 
                                help='Duration in seconds to record')
    recording_parser.add_argument('--output', type=str, default='voice_test_recording.wav', 
                                help='Output file path')
    
    # Transcription test
    transcription_parser = subparsers.add_parser('transcribe', help='Test speech transcription')
    transcription_parser.add_argument('--file', type=str, 
                                    help='Audio file to transcribe (optional)')
    
    # Model test
    subparsers.add_parser('model', help='Test speaker verification model loading')
    
    # Enrollment test
    enrollment_parser = subparsers.add_parser('enroll', help='Test user enrollment')
    enrollment_parser.add_argument('--name', type=str, default="Test User", 
                                 help='Name for the test user')
    enrollment_parser.add_argument('--samples', type=int, default=2, 
                                 help='Number of voice samples to record')
    
    # Verification test
    subparsers.add_parser('verify', help='Test speaker verification')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.test_type == 'record':
        success, _ = test_voice_recording(duration=args.duration, output_path=args.output)
        
    elif args.test_type == 'transcribe':
        success, text = test_speech_transcription(audio_path=args.file)
        
    elif args.test_type == 'model':
        success = test_speaker_verification_model()
        
    elif args.test_type == 'enroll':
        success = test_user_enrollment(user_name=args.name, num_samples=args.samples)
        
    elif args.test_type == 'verify':
        success, _ = test_speaker_verification()
        
    else:
        # Default: show help
        print("Voice Recognition Testing Utility")
        print("\nAvailable commands:")
        print("  python -m tests.test_voice_recognition record [--duration SECONDS] [--output PATH]")
        print("  python -m tests.test_voice_recognition transcribe [--file PATH]")
        print("  python -m tests.test_voice_recognition model")
        print("  python -m tests.test_voice_recognition enroll [--name \"User Name\"] [--samples COUNT]")
        print("  python -m tests.test_voice_recognition verify")
        parser.print_help()


if __name__ == "__main__":
    main() 