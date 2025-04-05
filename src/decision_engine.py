import yaml
import os
from src.face_recognition_module import FaceRecognizer
from src.voice_recognition_module import VoiceRecognizer


class DecisionEngine:
    def __init__(self, config_path='config/settings.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize face and voice recognizers
        self.face_recognizer = FaceRecognizer(config_path)
        self.voice_recognizer = VoiceRecognizer(config_path)
    
    def enroll_user(self, user_name, face_samples=5, voice_samples=1, delay=1):
        """
        Enroll a new user with both face and voice.
        
        Args:
            user_name (str): Name of the user to enroll
            face_samples (int): Number of face samples to capture
            voice_samples (int): Number of voice samples to record (default: 1 - only used for compatibility)
            delay (int): Delay between captures in seconds
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        print(f"Starting enrollment process for {user_name}")
        
        # First enroll face
        print("\n=== Face Enrollment ===")
        face_success = self.face_recognizer.enroll_user(
            user_name, num_samples=face_samples, delay=delay
        )
        
        if not face_success:
            print("Face enrollment failed. Aborting enrollment.")
            return False
        
        # Then enroll voice
        print("\n=== Voice Enrollment ===")
        voice_success = self.voice_recognizer.enroll_user(
            user_name, num_samples=1, delay=delay
        )
        
        if not voice_success:
            print("Voice enrollment failed. Enrollment incomplete.")
            return False
        
        print(f"\nUser {user_name} enrolled successfully with both face and voice.")
        return True
    
    def authenticate_user(self, face_time=5, display_video=False):
        """
        Authenticate a user using both face and voice recognition.
        
        Args:
            face_time (int): Time in seconds to attempt face recognition
            display_video (bool): Whether to display the video feed
            
        Returns:
            tuple: (authenticated, user_name) - whether authentication was successful and user name
        """
        # Step 1: Face Recognition
        print("\n=== Face Recognition ===")
        face_recognized, face_name = self.face_recognizer.recognize_face(
            display_video=display_video, recognition_time=face_time
        )
        
        if not face_recognized:
            print("Face not recognized. Authentication failed.")
            return False, None
        
        print(f"Face recognized as: {face_name}")
        
        # Step 2: Voice Verification
        print("\n=== Voice Verification ===")
        print(f"Please say something to verify your identity as {face_name}...")
        voice_verified, voice_name = self.voice_recognizer.verify_speaker()
        
        if not voice_verified:
            print("Voice not verified. Authentication failed.")
            return False, None
        
        print(f"Voice verified as: {voice_name}")
        
        # Step 3: Check if face and voice match the same person
        if face_name != voice_name:
            print(f"Face ({face_name}) and voice ({voice_name}) don't match. Authentication failed.")
            return False, None
        
        # Authentication successful
        print(f"\nAuthentication successful. Welcome, {face_name}!")
        return True, face_name
    
    def get_user_speech_input(self):
        """
        Get speech input from the user and transcribe it.
        
        Returns:
            str: Transcribed text or None if failed
        """
        return self.voice_recognizer.transcribe_speech()


if __name__ == "__main__":
    # Simple test
    engine = DecisionEngine()
    
    # Check if there are enrolled users
    if not engine.face_recognizer.known_face_names:
        print("No enrolled users found. Starting enrollment...")
        engine.enroll_user("Test User")
    
    # Authenticate
    authenticated, user = engine.authenticate_user(display_video=True)
    
    if authenticated:
        print(f"User {user} authenticated successfully!")
        
        # Get speech input
        print("\nPlease say something for transcription:")
        text = engine.get_user_speech_input()
        if text:
            print(f"You said: {text}")
    else:
        print("Authentication failed.")
