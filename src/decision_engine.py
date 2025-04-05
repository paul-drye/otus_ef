import yaml
import os
import cv2
import time
import numpy as np
import threading
import queue
import speech_recognition as sr
from face_recognition_module import FaceRecognizer
from voice_recognition_module import VoiceRecognizer


class DecisionEngine:
    def __init__(self, config_path='config/settings.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize face and voice recognizers
        self.face_recognizer = FaceRecognizer(config_path)
        self.voice_recognizer = VoiceRecognizer(config_path)
        
        # Authentication states
        self.is_authenticated = False
        self.authenticated_user = None
        
        # Authentication parameters
        self.face_check_interval = 0.5  # seconds
        self.min_faces_required = 3  # minimum number of face detections before recognition
        self.voice_silence_timeout = 1.0  # seconds of silence to end recording
        
        # Audio parameters
        self.sample_rate = self.config['voice_recognition']['sample_rate']
        self.chunk_size = 1024
        self.silence_threshold = 4000  # Lower threshold for better sensitivity (was 1000)
        
        # Threads and queues
        self.face_thread = None
        self.voice_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.stop_threads = False
        
        # Callbacks
        self.on_face_detected = None
        self.on_face_recognized = None
        self.on_voice_processed = None
        self.on_authentication_complete = None
        self.on_enrollment_required = None
    
    def set_callbacks(self, on_face_detected=None, on_face_recognized=None, 
                      on_voice_processed=None, on_authentication_complete=None,
                      on_enrollment_required=None):
        """
        Set callback functions for authentication events.
        
        Args:
            on_face_detected: Called when a face is detected in the video
            on_face_recognized: Called when a face is successfully recognized
            on_voice_processed: Called when voice has been processed
            on_authentication_complete: Called when authentication is complete
            on_enrollment_required: Called when enrollment is required
        """
        self.on_face_detected = on_face_detected
        self.on_face_recognized = on_face_recognized
        self.on_voice_processed = on_voice_processed
        self.on_authentication_complete = on_authentication_complete
        self.on_enrollment_required = on_enrollment_required
    
    def enroll_user(self, user_name, face_samples=5, voice_samples=1, delay=1, confirm=True):
        """
        Enroll a new user with both face and voice.
        
        Args:
            user_name (str): Name of the user to enroll
            face_samples (int): Number of face samples to capture
            voice_samples (int): Number of voice samples to record
            delay (int): Delay between captures in seconds
            confirm (bool): Whether to ask for confirmation during enrollment
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        print(f"Starting enrollment process for {user_name}")
        
        # First enroll face
        print("\n=== Face Enrollment ===")
        face_success = self.face_recognizer.enroll_user(
            user_name, num_samples=face_samples, delay=delay, confirm=confirm
        )
        
        if not face_success:
            print("Face enrollment failed. Aborting enrollment.")
            return False
        
        # Then enroll voice
        print("\n=== Voice Enrollment ===")
        voice_success = self.voice_recognizer.enroll_user(
            user_name, num_samples=1, delay=delay, confirm=confirm
        )
        
        if not voice_success:
            print("Voice enrollment failed. Enrollment incomplete.")
            return False
        
        print(f"\nUser {user_name} enrolled successfully with both face and voice.")
        return True

    def _face_detection_worker(self):
        """Worker thread for continuous face detection and recognition."""
        import face_recognition  # Import here to avoid circular imports
        
        if not self.face_recognizer._initialize_camera():
            print("Failed to initialize camera")
            return

        face_detections = []
        last_recognition_time = 0
        recognized_user = None
        
        while not self.stop_threads:
            ret, frame = self.face_recognizer.camera.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Resize frame for face detection
            small_frame = cv2.resize(frame, (0, 0), 
                                    fx=self.face_recognizer.scale_factor, 
                                    fy=self.face_recognizer.scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Store the original frame and face locations
            if face_locations:
                face_detections.append((frame.copy(), face_locations))
                
                # Keep only the most recent faces
                if len(face_detections) > self.min_faces_required:
                    face_detections.pop(0)
                    
                # Notify face detected (without print statements)
                if self.on_face_detected:
                    self.on_face_detected(frame, face_locations)
                
                # Check if we have enough face detections and time for recognition
                current_time = time.time()
                if (len(face_detections) >= self.min_faces_required and 
                    current_time - last_recognition_time > self.face_check_interval):
                    
                    # Process the latest frame for recognition
                    latest_frame, latest_locations = face_detections[-1]
                    
                    # Get face encodings for the detected faces
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    if face_encodings:
                        # Compare with known face encodings
                        matches = []
                        for face_encoding in face_encodings:
                            if not self.face_recognizer.known_face_encodings:
                                break
                                
                            # Compare faces
                            matches = face_recognition.compare_faces(
                                self.face_recognizer.known_face_encodings, 
                                face_encoding,
                                tolerance=self.face_recognizer.tolerance
                            )
                            
                            # If match found
                            if True in matches:
                                matched_idx = matches.index(True)
                                recognized_user = self.face_recognizer.known_face_names[matched_idx]
                                
                                # Only notify once when user changes
                                if self.on_face_recognized:
                                    self.on_face_recognized(recognized_user)
                                
                                # Start voice verification
                                self._start_voice_verification(recognized_user)
                                
                                # Update recognition time
                                last_recognition_time = current_time
                                break
                        
                        # If no match, suggest enrollment once
                        if True not in matches and self.face_recognizer.known_face_encodings:
                            if self.on_enrollment_required:
                                self.on_enrollment_required("face")
            
            # Display frame with face locations
            if self.face_recognizer.known_face_names:
                for top, right, bottom, left in face_locations:
                    # Scale back up face locations
                    top = int(top / self.face_recognizer.scale_factor)
                    right = int(right / self.face_recognizer.scale_factor)
                    bottom = int(bottom / self.face_recognizer.scale_factor)
                    left = int(left / self.face_recognizer.scale_factor)
                    
                    # Draw rectangle around the face
                    color = (0, 255, 0) if recognized_user else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label with name
                    label = recognized_user or "Unknown"
                    cv2.putText(frame, label, (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display the resulting frame
            cv2.imshow('Face Recognition', frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.01)  # Small delay to reduce CPU usage
            
        # Release resources
        cv2.destroyAllWindows()
    
    def _audio_capture_worker(self):
        """Worker thread for continuous audio capture."""
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=self.sample_rate,
                          input=True,
                          input_device_index=self.voice_recognizer.audio_device_index,
                          frames_per_buffer=self.chunk_size)
            
            print("Audio capture started")
            
            while not self.stop_threads:
                try:
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_data)
                except Exception as e:
                    print(f"Error capturing audio: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Error setting up audio stream: {e}")
        
        finally:
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
            print("Audio capture stopped")
    
    def _voice_processing_worker(self, recognized_user):
        """
        Worker thread for voice activation detection and verification.
        
        Args:
            recognized_user (str): The user recognized by face recognition
        """
        import wave
        import numpy as np
        from array import array
        import tempfile
        
        # Clear the audio queue to start fresh
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Parameters for voice detection - adjust these for better sensitivity
        is_speaking = False
        silent_frames = 0
        speech_frames = []
        
        # Lower threshold for better voice detection
        voice_threshold = self.silence_threshold * 0.7
        
        # Debug counters
        total_frames = 0
        voiced_frames = 0
        
        # Temporary file for audio
        temp_audio_file = os.path.join(
            self.voice_recognizer.voice_data_path, 
            f"temp_verification_{int(time.time())}.wav"
        )
        
        # Store raw audio levels for debugging
        audio_levels = []
        
        try:
            while not self.stop_threads and total_frames < 1000:  # Limit total frames to prevent infinite loop
                if self.audio_queue.empty():
                    time.sleep(0.05)
                    continue
                
                # Get audio data from queue
                audio_data = self.audio_queue.get()
                total_frames += 1
                
                # Convert audio data to numpy array for analysis
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Check if audio level exceeds threshold (voice activity detection)
                audio_level = np.abs(audio_array).mean()
                audio_levels.append(audio_level)
                
                if not is_speaking and audio_level > voice_threshold:
                    # Start of speech detected
                    is_speaking = True
                    silent_frames = 0
                    voiced_frames = 1
                
                if is_speaking:
                    # Add audio data to speech frames
                    speech_frames.append(audio_data)
                    
                    # Check for silence
                    if audio_level < voice_threshold:
                        silent_frames += 1
                        # Convert silent frames to time
                        silent_time = silent_frames * self.chunk_size / self.sample_rate
                        
                        if silent_time >= self.voice_silence_timeout:
                            # End of speech detected
                            
                            # Only process if we have enough voiced frames
                            if voiced_frames > 10:  # Minimum frames to consider as valid speech
                                # Save audio data to temporary file
                                with wave.open(temp_audio_file, 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)  # 2 bytes for int16
                                    wf.setframerate(self.sample_rate)
                                    wf.writeframes(b''.join(speech_frames))
                                
                                # Verify speaker
                                verified, verified_user, confidence = self.voice_recognizer.verify_speaker(temp_audio_file)
                                
                                # Notify voice processed
                                if self.on_voice_processed:
                                    self.on_voice_processed(verified, verified_user)
                                
                                # Check authentication
                                if verified and verified_user == recognized_user:
                                    # Authentication successful
                                    self.is_authenticated = True
                                    self.authenticated_user = verified_user
                                    
                                    if self.on_authentication_complete:
                                        self.on_authentication_complete(True, verified_user)
                                
                                elif verified:
                                    # Voice verified but doesn't match face
                                    if self.on_authentication_complete:
                                        self.on_authentication_complete(False, None)
                                
                                else:
                                    # Voice not verified
                                    if self.on_enrollment_required:
                                        self.on_enrollment_required("voice")
                                
                                # Cleanup
                                if os.path.exists(temp_audio_file):
                                    os.remove(temp_audio_file)
                            
                            # Reset for next speech
                            is_speaking = False
                            speech_frames = []
                            silent_frames = 0
                            voiced_frames = 0
                            break  # Exit after processing one speech segment
                    else:
                        # Still speaking, reset silent counter
                        silent_frames = 0
                        voiced_frames += 1
        
        except Exception as e:
            print(f"Error in voice processing: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
    
    def _start_voice_verification(self, recognized_user):
        """Start voice verification after face is recognized."""
        # Stop previous voice thread if running
        if self.voice_thread and self.voice_thread.is_alive():
            return  # Already running
        
        # Start voice processing for the recognized user
        self.voice_thread = threading.Thread(
            target=self._voice_processing_worker, 
            args=(recognized_user,)
        )
        self.voice_thread.daemon = True
        self.voice_thread.start()
    
    def start_continuous_authentication(self):
        """Start continuous authentication process."""
        print("Starting continuous authentication...")
        
        # Ensure camera is initialized
        if not self.face_recognizer._initialize_camera():
            print("Failed to initialize camera")
            return False
        
        # Reset state
        self.stop_threads = False
        self.is_authenticated = False
        self.authenticated_user = None
        
        # Start audio capture thread
        audio_thread = threading.Thread(target=self._audio_capture_worker)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Start face detection thread
        self.face_thread = threading.Thread(target=self._face_detection_worker)
        self.face_thread.daemon = True
        self.face_thread.start()
        
        return True
    
    def stop_continuous_authentication(self):
        """Stop the continuous authentication process."""
        print("Stopping continuous authentication...")
        self.stop_threads = True
        
        # Wait for threads to complete
        if self.face_thread and self.face_thread.is_alive():
            self.face_thread.join(timeout=2)
        
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2)
        
        # Release camera
        self.face_recognizer._release_camera()
        
        # Clear resources
        self.is_authenticated = False
        self.authenticated_user = None
        
        print("Continuous authentication stopped")
        return True
    
    def get_user_speech_input(self):
        """
        Get speech input from the user and transcribe it.
        
        Returns:
            str: Transcribed text or None if failed
        """
        return self.voice_recognizer.transcribe_speech()


if __name__ == "__main__":
    # Simple test of continuous authentication
    engine = DecisionEngine()
    
    def on_face_detected(frame, face_locations):
        print(f"Face detected: {len(face_locations)} faces")
    
    def on_face_recognized(user):
        print(f"Face recognized: {user}")
    
    def on_voice_processed(verified, user):
        print(f"Voice processed: {'verified' if verified else 'not verified'} - {user}")
    
    def on_authentication_complete(success, user):
        if success:
            print(f"Authentication successful for user: {user}")
        else:
            print("Authentication failed")
    
    def on_enrollment_required(modality):
        print(f"Enrollment required for {modality}")
    
    # Set callbacks
    engine.set_callbacks(
        on_face_detected=on_face_detected,
        on_face_recognized=on_face_recognized,
        on_voice_processed=on_voice_processed,
        on_authentication_complete=on_authentication_complete,
        on_enrollment_required=on_enrollment_required
    )
    
    # Start continuous authentication
    engine.start_continuous_authentication()
    
    try:
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop_continuous_authentication()
