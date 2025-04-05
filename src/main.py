#!/usr/bin/env python3
import os
import sys
import argparse
import time
import queue
import speech_recognition as sr

from decision_engine import DecisionEngine
from robot_interaction import RobotInteraction


def create_user(decision_engine, robot, args):
    """Enroll a new user with face and voice."""
    robot.speak("Starting user enrollment process.")
    robot.enrollment_instruction("face")
    
    time.sleep(1)
    success = decision_engine.enroll_user(
        args.name, 
        face_samples=args.face_samples, 
        voice_samples=args.voice_samples,
        confirm=not args.no_confirm
    )
    
    if success:
        robot.enrollment_success()
    else:
        robot.enrollment_failed()
    
    return success


def authenticate(decision_engine, robot, args):
    """Authenticate a user."""
    robot.speak("Starting authentication process.")
    
    # Prompt for face
    robot.prompt_face()
    time.sleep(1)
    
    # Authenticate
    authenticated, user_name = decision_engine.authenticate_user(
        face_time=args.face_time, display_video=args.display_video
    )
    
    if authenticated:
        robot.authentication_success(user_name)
        return True, user_name
    else:
        robot.authentication_failed()
        return False, None


def continuous_authentication(decision_engine, robot, args):
    """Run continuous authentication with face detection followed by voice activation."""
    print("\n=== Continuous Authentication Mode ===")
    print("Looking for faces in the video feed...")
    
    # Override silence threshold if specified
    if args.threshold is not None:
        decision_engine.silence_threshold = args.threshold
        print(f"Setting audio threshold to {args.threshold} (lower = more sensitive)")
    
    # Run audio debug if requested
    if args.debug_audio:
        print("\n=== Audio Debug Mode ===")
        print("Testing microphone input. Speak to see audio levels.")
        print("This will help determine if your microphone is working properly.")
        print("Press Ctrl+C to stop the test and continue.")
        
        import pyaudio
        import numpy as np
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=decision_engine.sample_rate,
                          input=True,
                          input_device_index=decision_engine.voice_recognizer.audio_device_index,
                          frames_per_buffer=decision_engine.chunk_size)
            
            print("Audio capture started")
            print(f"Current threshold: {decision_engine.silence_threshold} (lower = more sensitive)")
            print(f"Recommended range: 200-800 depending on your microphone and environment")
            print("You should see levels rise above the threshold when speaking")
            
            while True:
                try:
                    audio_data = stream.read(decision_engine.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.abs(audio_array).mean()
                    
                    # Create a simple visual meter
                    meter = "█" * int(audio_level / 50)
                    threshold_marker = "▓" * int(decision_engine.silence_threshold / 50)
                    
                    is_speech = "SPEECH" if audio_level > decision_engine.silence_threshold else "silence"
                    print(f"Level: {audio_level:6.1f} {meter}")
                    print(f"Threshold: {decision_engine.silence_threshold:6.1f} {threshold_marker} ({is_speech})")
                    print("\033[2A", end="")  # Move cursor up 2 lines
                    
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error capturing audio: {e}")
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nAudio test complete. Continuing with authentication.")
        
        finally:
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    # Message queue for thread-safe communication
    message_queue = queue.Queue()
    
    # Current recognized user
    current_user = None
    
    # Set up callbacks
    def on_face_detected(frame, face_locations):
        if args.verbose:
            print(f"Face detected: {len(face_locations)} faces")
    
    def on_face_recognized(user):
        nonlocal current_user
        if user != current_user:  # Only print when user changes
            print(f"\nFace recognized: {user}")
            current_user = user
    
    def on_voice_processed(verified, user):
        if verified:
            # Only print when verification succeeds
            print(f"Voice verified as: {user}")
    
    def on_authentication_complete(success, user):
        if success:
            print(f"\nAuthentication successful for user: {user}")
            message_queue.put(("auth_success", user))
        else:
            # Always go to interactive mode even if authentication fails
            message_queue.put(("start_interactive", current_user or "Unknown Person"))
    
    def on_enrollment_required(modality):
        # Always go to interactive mode even if enrollment is needed
        message_queue.put(("start_interactive", current_user or "Unknown Person"))
    
    # Set callbacks
    decision_engine.set_callbacks(
        on_face_detected=on_face_detected,
        on_face_recognized=on_face_recognized,
        on_voice_processed=on_voice_processed,
        on_authentication_complete=on_authentication_complete,
        on_enrollment_required=on_enrollment_required
    )
    
    # Start continuous authentication
    if not decision_engine.start_continuous_authentication():
        print("Failed to start authentication. Please check your camera.")
        return False
    
    try:
        print("Press Ctrl+C to stop")
        
        # Main loop - wait for authentication
        while True:
            # Process any messages from threads
            try:
                message_type, data = message_queue.get(block=False)
                
                # Handle messages from threads
                if message_type == "auth_success":
                    # Start interactive mode with authenticated user
                    interactive_session(decision_engine, data)
                    
                    # Reset authentication state after interactive session
                    decision_engine.is_authenticated = False
                    decision_engine.authenticated_user = None
                    print("\nResuming authentication...")
                
                elif message_type == "start_interactive":
                    # Start interactive mode with unauthenticated user
                    interactive_session(decision_engine, data)
                    
                    # Reset authentication state after interactive session
                    decision_engine.is_authenticated = False
                    decision_engine.authenticated_user = None
                    print("\nResuming authentication...")
                
            except queue.Empty:
                pass  # No messages to process
            
            # If authenticated but not yet processed
            if decision_engine.is_authenticated and decision_engine.authenticated_user:
                user = decision_engine.authenticated_user
                
                # Start interactive mode with authenticated user
                interactive_session(decision_engine, user)
                
                # Reset authentication state after interactive session
                decision_engine.is_authenticated = False
                decision_engine.authenticated_user = None
                print("\nResuming authentication...")
            
            time.sleep(0.1)  # Shorter sleep time for more responsive UI
            
    except KeyboardInterrupt:
        print("\nStopping continuous authentication")
    finally:
        decision_engine.stop_continuous_authentication()
    
    return True


def interactive_session(decision_engine, user_name):
    """Run interactive session with a user (authenticated or not)."""
    print(f"\n=== Interactive Mode ({user_name}) ===")
    print("Say something or 'exit' to end the session.")
    
    # Keep track of the expected speaker
    authenticated_name = user_name if user_name != "Unknown Person" else None
    
    # Set up VAD parameters
    chunk_size = decision_engine.chunk_size
    sample_rate = decision_engine.sample_rate
    silence_threshold = decision_engine.silence_threshold * 0.7  # Same as in voice processing worker
    silence_timeout = decision_engine.voice_silence_timeout
    
    import pyaudio
    import wave
    import numpy as np
    import os
    
    # Create a PyAudio instance to be reused
    p = pyaudio.PyAudio()
    
    try:
        while True:
            print("\nListening for speech...")
            
            # Create a temporary file for the recording
            temp_audio_file = os.path.join(
                decision_engine.voice_recognizer.voice_data_path, 
                f"temp_interactive_{int(time.time())}.wav"
            )
            
            # Use VAD to capture speech
            try:
                # Open audio stream
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=sample_rate,
                                input=True,
                                input_device_index=decision_engine.voice_recognizer.audio_device_index,
                                frames_per_buffer=chunk_size)
                
                # Variables for voice detection
                is_speaking = False
                silent_frames = 0
                speech_frames = []
                voiced_frames = 0
                total_frames = 0
                
                # Recording state
                recording_timeout = 500  # Maximum frames to record (about 10 seconds)
                
                # Main recording loop
                while total_frames < recording_timeout:
                    # Read audio data
                    audio_data = stream.read(chunk_size, exception_on_overflow=False)
                    total_frames += 1
                    
                    # Convert to numpy array for analysis
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.abs(audio_array).mean()
                    
                    if not is_speaking and audio_level > silence_threshold:
                        # Start of speech detected
                        is_speaking = True
                        silent_frames = 0
                        voiced_frames = 1
                        speech_frames = [audio_data]  # Start collecting frames
                        
                        # Visual indicator
                        print("Speech detected, recording...", end="\r")
                    
                    elif is_speaking:
                        # Continue recording speech
                        speech_frames.append(audio_data)
                        
                        # Check if still speaking
                        if audio_level > silence_threshold:
                            silent_frames = 0
                            voiced_frames += 1
                        else:
                            silent_frames += 1
                            
                            # Calculate silent time
                            silent_time = silent_frames * chunk_size / sample_rate
                            
                            # End of speech detection
                            if silent_time >= silence_timeout:
                                print("End of speech detected   ")
                                break
                
                # Close the stream
                stream.stop_stream()
                stream.close()
                
                # Process the collected speech if we have enough frames
                if is_speaking and voiced_frames > 10:  # Minimum frames for valid speech
                    # Save the audio to a temporary file
                    with wave.open(temp_audio_file, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(sample_rate)
                        wf.writeframes(b''.join(speech_frames))
                    
                    # First verify the speaker if authenticated
                    speaker_verified = False
                    verified_name = None
                    confidence_score = 0.0
                    
                    if authenticated_name and decision_engine.voice_recognizer.model_loaded:
                        verified, verified_name, confidence_score = decision_engine.voice_recognizer.verify_speaker(temp_audio_file)
                        speaker_verified = verified and verified_name == authenticated_name
                    
                    # Then transcribe the speech
                    text = None
                    try:
                        with sr.AudioFile(temp_audio_file) as source:
                            audio_data = decision_engine.voice_recognizer.recognizer.record(source)
                            text = decision_engine.voice_recognizer.recognizer.recognize_google(audio_data)
                    except Exception as e:
                        print(f"Error transcribing speech: {e}")
                    
                    # Clean up temporary file
                    if os.path.exists(temp_audio_file):
                        os.remove(temp_audio_file)
                    
                    if not text:
                        print("Couldn't understand audio. Please try again.")
                        continue
                    
                    # Display user speech with verification status
                    if authenticated_name:
                        if speaker_verified:
                            print(f"{user_name} [{confidence_score:.2f}] said: \"{text}\"")
                        else:
                            if verified_name:
                                print(f"WARNING: Voice detected as {verified_name} [{confidence_score:.2f}], not {authenticated_name}")
                                print(f"Unverified speaker said: \"{text}\"")
                            else:
                                print(f"WARNING: Unverified speaker [{confidence_score:.2f} < {decision_engine.voice_recognizer.verification_threshold}] said: \"{text}\"")
                    else:
                        # For unauthenticated users, just show the text
                        print(f"{user_name} said: \"{text}\"")
                    
                    # Check for exit command
                    if 'exit' in text.lower() or 'quit' in text.lower():
                        print("Exiting interactive mode.")
                        break
                else:
                    # Not enough speech detected
                    print("Not enough speech detected, please try again.")
            
            except Exception as e:
                print(f"Error during audio capture: {e}")
                import traceback
                traceback.print_exc()
                # Clean up if needed
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
    
    finally:
        # Clean up PyAudio
        p.terminate()


def interactive_mode(decision_engine, robot):
    """Run in interactive mode, allowing command input after authentication."""
    authenticated, user_name = authenticate(decision_engine, robot, parse_args(['auth']))
    
    if not authenticated:
        return
    
    robot.speak("You are now in interactive mode. Speak commands or say 'exit' to quit.")
    
    while True:
        # Get speech input
        print("\nListening for command...")
        command = decision_engine.get_user_speech_input()
        
        if not command:
            robot.speak("I didn't catch that. Could you please repeat?")
            continue
        
        print(f"Command: {command}")
        
        # Check for exit command
        if 'exit' in command.lower() or 'quit' in command.lower():
            robot.speak("Exiting interactive mode. Goodbye!")
            break
        
        # Try to respond to the command
        if not robot.respond_to_command(command):
            robot.speak("I'm not sure how to respond to that command.")


def list_users(decision_engine, robot):
    """List all enrolled users."""
    face_users = decision_engine.face_recognizer.known_face_names
    voice_users = list(decision_engine.voice_recognizer.speaker_embeddings.keys())
    
    print("\n=== Enrolled Users ===")
    print("Face recognition users:")
    for user in face_users:
        print(f"- {user}")
    
    print("\nVoice recognition users:")
    for user in voice_users:
        print(f"- {user}")
    
    # Find users enrolled with both face and voice
    common_users = set(face_users).intersection(set(voice_users))
    print("\nFully enrolled users (face + voice):")
    for user in common_users:
        print(f"- {user}")
    
    if common_users:
        robot.speak(f"There are {len(common_users)} fully enrolled users in the system.")
    else:
        robot.speak("There are no fully enrolled users in the system.")


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face and Voice Authentication System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', help='Enroll a new user')
    enroll_parser.add_argument('--name', '-n', required=True, help='Name of the user to enroll')
    enroll_parser.add_argument('--face-samples', type=int, default=5, 
                               help='Number of face samples to capture')
    enroll_parser.add_argument('--voice-samples', type=int, default=1, 
                               help='Number of voice samples to record (always 1, kept for backward compatibility)')
    enroll_parser.add_argument('--no-confirm', action='store_true',
                               help='Skip confirmation steps during enrollment')
    
    # Authenticate command
    auth_parser = subparsers.add_parser('auth', help='Authenticate a user')
    auth_parser.add_argument('--face-time', type=int, default=5, 
                            help='Time in seconds to attempt face recognition')
    auth_parser.add_argument('--display-video', action='store_true', 
                            help='Display video feed during face recognition')
    
    # Continuous authentication command
    continuous_parser = subparsers.add_parser('continuous', help='Run continuous authentication mode')
    continuous_parser.add_argument('--auto-enroll', action='store_true',
                                 help='Automatically prompt for enrollment if authentication fails')
    continuous_parser.add_argument('--verbose', action='store_true',
                                 help='Print detailed status messages')
    continuous_parser.add_argument('--debug-audio', action='store_true',
                                 help='Run audio input test before starting authentication')
    continuous_parser.add_argument('--threshold', type=int, default=None,
                                 help='Override the audio silence threshold (lower = more sensitive)')
    
    # Interactive mode command
    subparsers.add_parser('interactive', help='Run in interactive mode after authentication')
    
    # List users command
    subparsers.add_parser('list-users', help='List all enrolled users')
    
    return parser.parse_args(args)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create decision engine and robot interaction
    decision_engine = DecisionEngine()
    robot = RobotInteraction()
    
    if args.command == 'enroll':
        create_user(decision_engine, robot, args)
    elif args.command == 'auth':
        authenticate(decision_engine, robot, args)
    elif args.command == 'continuous':
        continuous_authentication(decision_engine, robot, args)
    elif args.command == 'interactive':
        interactive_mode(decision_engine, robot)
    elif args.command == 'list-users':
        list_users(decision_engine, robot)
    else:
        # Display help if no command is provided
        print("Please specify a command.")
        robot.speak("Please specify a command such as enroll, auth, continuous, or interactive.")


if __name__ == "__main__":
    main()
