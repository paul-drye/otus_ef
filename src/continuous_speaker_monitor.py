import os
import numpy as np
import torch
import time
import threading
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque

from audio_stream import AudioStream
from buffer_manager import BufferManager
from voice_activity_detector import VoiceActivityDetector
from voice_recognition_module import VoiceRecognizer


class VerificationResult:
    """Stores speaker verification results with confidence tracking."""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize verification result tracker.
        
        Args:
            window_size: Number of results to track for confidence calculation
        """
        self.speaker_name = None
        self.confidence = 0.0
        self.last_scores = deque(maxlen=window_size)
        self.consecutive_matches = 0
        self.consecutive_mismatches = 0
        self.last_update_time = 0.0
    
    def update(self, speaker_name: Optional[str], score: float) -> None:
        """
        Update verification result.
        
        Args:
            speaker_name: Name of verified speaker, or None if not verified
            score: Confidence score
        """
        self.last_update_time = time.time()
        self.last_scores.append(score)
        
        if speaker_name is not None:
            if self.speaker_name == speaker_name:
                # Same speaker, increase consecutive matches
                self.consecutive_matches += 1
                self.consecutive_mismatches = 0
            else:
                # Different speaker
                self.consecutive_mismatches += 1
                
                # Only change speaker name after consistent mismatches
                if self.consecutive_mismatches >= 3:
                    self.speaker_name = speaker_name
                    self.consecutive_matches = 1
                    self.consecutive_mismatches = 0
        else:
            # No speaker detected
            self.consecutive_mismatches += 1
            
            # Reset after consistent mismatches
            if self.consecutive_mismatches >= 5:
                self.speaker_name = None
                self.consecutive_matches = 0
        
        # Update confidence score (average of recent scores)
        if self.last_scores:
            self.confidence = sum(self.last_scores) / len(self.last_scores)
    
    def is_verified(self) -> bool:
        """Check if a speaker is verified with sufficient confidence."""
        return self.speaker_name is not None and self.consecutive_matches >= 2
    
    def get_speaker(self) -> Optional[str]:
        """Get the verified speaker name, or None if not verified."""
        return self.speaker_name if self.is_verified() else None
    
    def get_confidence(self) -> float:
        """Get the current confidence score."""
        return self.confidence
    
    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if the result is stale (not updated recently)."""
        return time.time() - self.last_update_time > max_age_seconds


class ContinuousSpeakerMonitor:
    """
    Continuously monitors audio for speaker verification and speech commands.
    
    Integrates:
    - Audio streaming
    - Buffer management
    - Voice activity detection
    - Speaker verification
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 device_index: Optional[int] = None,
                 verification_window_seconds: float = 2.0,
                 verification_interval_seconds: float = 0.5):
        """
        Initialize the continuous speaker monitor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            device_index: Audio device index to use
            verification_window_seconds: Size of verification window in seconds
            verification_interval_seconds: Time between verification attempts in seconds
        """
        self.sample_rate = sample_rate
        self.verification_interval_seconds = verification_interval_seconds
        
        # Create components
        self.audio_stream = AudioStream(
            sample_rate=sample_rate,
            chunk_size=1024,
            device_index=device_index
        )
        
        self.buffer_manager = BufferManager(
            sample_rate=sample_rate,
            verification_window_seconds=verification_window_seconds,
            verification_step_seconds=verification_interval_seconds
        )
        
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            threshold=0.5,
            neg_threshold=0.3,
            activation_window=5,
            deactivation_window=20
        )
        
        self.voice_recognizer = VoiceRecognizer()
        
        # Verification state
        self.verification_result = VerificationResult()
        self.last_verification_time = 0.0
        
        # Speech processing state
        self.is_command_active = False
        self.verification_active = True
        
        # Callbacks
        self.on_speaker_change_callbacks = []
        self.on_speech_command_callbacks = []
        
        # Processing flags
        self.is_running = False
        self.verification_thread = None
        self.do_verification = False
        
        # Set up VAD callbacks
        self.vad.add_speech_start_callback(self._on_speech_start)
        self.vad.add_speech_end_callback(self._on_speech_end)
    
    def _audio_callback(self, audio_data: np.ndarray) -> None:
        """
        Process incoming audio data.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Process audio through buffer manager
        status = self.buffer_manager.process_audio_chunk(audio_data)
        
        # Process audio through VAD
        is_speech, speech_prob = self.vad.process_audio(audio_data)
        
        # Get verification window if ready and it's time to verify
        if (status["verification_ready"] and 
            status["verification_window"] is not None and 
            self.verification_active):
            
            # Check if it's time for verification
            current_time = time.time()
            if current_time - self.last_verification_time >= self.verification_interval_seconds:
                self.last_verification_time = current_time
                
                # Don't block the audio processing thread with verification
                # Instead, queue verification for a separate thread
                self.verification_data = status["verification_window"]
                self.do_verification = True
    
    def _verification_worker(self) -> None:
        """Worker thread for speaker verification."""
        while self.is_running:
            # Check if verification is needed
            if self.do_verification and self.verification_data is not None:
                try:
                    # Extract verification data
                    verification_audio = self.verification_data
                    self.verification_data = None
                    self.do_verification = False
                    
                    # Only verify if speech is detected with sufficient energy
                    if np.abs(verification_audio).mean() > 0.01:
                        # Perform verification
                        verified, name = self.voice_recognizer.verify_speaker_from_audio(verification_audio)
                        
                        # Get verification score
                        score = 0.0
                        if hasattr(self.voice_recognizer, 'last_verification_score'):
                            score = self.voice_recognizer.last_verification_score
                        
                        # Update verification result
                        prev_speaker = self.verification_result.get_speaker()
                        self.verification_result.update(name if verified else None, score)
                        current_speaker = self.verification_result.get_speaker()
                        
                        # Notify if speaker changed
                        if prev_speaker != current_speaker:
                            self._handle_speaker_change(prev_speaker, current_speaker)
                    
                except Exception as e:
                    print(f"Error in verification worker: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Sleep to avoid busy waiting
            time.sleep(0.1)
    
    def _on_speech_start(self) -> None:
        """Handle speech start events from VAD."""
        if not self.is_command_active:
            print("Speech detected, starting command capture")
            self.is_command_active = True
            self.buffer_manager.start_command()
    
    def _on_speech_end(self) -> None:
        """Handle speech end events from VAD."""
        if self.is_command_active:
            print("Speech ended, finishing command capture")
            audio, metadata = self.buffer_manager.end_command()
            self.is_command_active = False
            
            # Process speech command if we have a verified speaker
            if len(audio) > 0 and self.verification_result.is_verified():
                self._process_speech_command(audio, metadata, self.verification_result.get_speaker())
    
    def _process_speech_command(self, 
                               audio_data: np.ndarray, 
                               metadata: Dict[str, Any],
                               speaker: str) -> None:
        """
        Process a speech command.
        
        Args:
            audio_data: Audio data for the speech command
            metadata: Metadata about the speech command
            speaker: Verified speaker name
        """
        # Transcribe the command
        try:
            transcript = self.voice_recognizer.transcribe_audio(audio_data)
            
            if transcript:
                command_info = {
                    "speaker": speaker,
                    "transcript": transcript,
                    "audio": audio_data,
                    "metadata": metadata,
                    "confidence": self.verification_result.get_confidence(),
                    "timestamp": time.time()
                }
                
                # Notify command listeners
                self._notify_command_listeners(command_info)
            else:
                print("Failed to transcribe command")
                
        except Exception as e:
            print(f"Error processing speech command: {e}")
    
    def _handle_speaker_change(self, 
                              previous_speaker: Optional[str], 
                              current_speaker: Optional[str]) -> None:
        """
        Handle speaker change events.
        
        Args:
            previous_speaker: Previous speaker name or None
            current_speaker: Current speaker name or None
        """
        if previous_speaker != current_speaker:
            change_info = {
                "previous_speaker": previous_speaker,
                "current_speaker": current_speaker,
                "confidence": self.verification_result.get_confidence(),
                "timestamp": time.time()
            }
            
            # Log the change
            if current_speaker:
                print(f"Speaker changed: {previous_speaker} -> {current_speaker}")
            else:
                print(f"Speaker lost: {previous_speaker} -> None")
            
            # Notify speaker change listeners
            self._notify_speaker_change_listeners(change_info)
    
    def _notify_speaker_change_listeners(self, change_info: Dict[str, Any]) -> None:
        """Notify all speaker change listeners."""
        for callback in self.on_speaker_change_callbacks:
            try:
                callback(change_info)
            except Exception as e:
                print(f"Error in speaker change callback: {e}")
    
    def _notify_command_listeners(self, command_info: Dict[str, Any]) -> None:
        """Notify all command listeners."""
        for callback in self.on_speech_command_callbacks:
            try:
                callback(command_info)
            except Exception as e:
                print(f"Error in command callback: {e}")
    
    def add_speaker_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback for speaker change events.
        
        Args:
            callback: Function taking a change_info dictionary
        """
        self.on_speaker_change_callbacks.append(callback)
    
    def add_speech_command_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback for speech command events.
        
        Args:
            callback: Function taking a command_info dictionary
        """
        self.on_speech_command_callbacks.append(callback)
    
    def get_current_speaker(self) -> Optional[str]:
        """Get the currently verified speaker, or None if none."""
        return self.verification_result.get_speaker()
    
    def start(self) -> None:
        """Start continuous speaker monitoring."""
        if self.is_running:
            print("Continuous speaker monitoring is already running")
            return
        
        # Reset state
        self.verification_result = VerificationResult()
        self.last_verification_time = 0.0
        self.is_command_active = False
        self.verification_data = None
        self.do_verification = False
        
        # Start verification thread
        self.is_running = True
        self.verification_thread = threading.Thread(target=self._verification_worker)
        self.verification_thread.daemon = True
        self.verification_thread.start()
        
        # Register audio callback
        self.audio_stream.register_callback(self._audio_callback)
        
        # Start audio stream
        self.audio_stream.start()
        
        print("Continuous speaker monitoring started")
    
    def stop(self) -> None:
        """Stop continuous speaker monitoring."""
        if not self.is_running:
            print("Continuous speaker monitoring is not running")
            return
        
        # Stop audio stream
        self.audio_stream.stop()
        
        # Stop verification thread
        self.is_running = False
        if self.verification_thread and self.verification_thread.is_alive():
            self.verification_thread.join(timeout=1.0)
        
        print("Continuous speaker monitoring stopped")


# Extend the VoiceRecognizer class to support direct audio verification
def extend_voice_recognizer():
    """Extend the VoiceRecognizer class with methods for direct audio processing."""
    
    # Add method to verify from audio data directly
    def verify_speaker_from_audio(self, audio_data):
        """
        Verify speaker from audio data directly.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (verified, name) - whether speaker is verified and their name
        """
        if not self.model_loaded or not self.speaker_embeddings:
            return False, None
        
        # Extract embedding from the audio data
        embedding = self.extract_embedding_from_audio(audio_data)
        
        if embedding is None:
            return False, None
        
        # Compare with enrolled speakers using the verification model's scoring
        best_match = None
        best_score = -float('inf')
        
        for name, stored_embedding in self.speaker_embeddings.items():
            try:
                # Convert embeddings if needed
                if isinstance(stored_embedding, np.ndarray):
                    stored_embedding = torch.from_numpy(stored_embedding)
                
                # Use the verification model's scoring function
                score = self.verification_model.similarity(embedding.unsqueeze(0), 
                                                          stored_embedding.unsqueeze(0))
                score = score.item()
                
                # Store for debugging and confidence metrics
                self.last_verification_score = score
                
                if score > best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error comparing with user {name}: {e}")
                continue
        
        # Verify if the best match is above the threshold
        if best_match and best_score > self.verification_threshold:
            return True, best_match
        else:
            return False, None
    
    # Add method to extract embedding directly from audio data
    def extract_embedding_from_audio(self, audio_data):
        """
        Extract speaker embedding directly from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            torch.Tensor: Speaker embedding
        """
        if not self.model_loaded:
            return None
        
        try:
            # Ensure audio is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is in the correct range [-1, 1]
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).float()
            
            # Reshape if needed
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Move to the same device as the model
            if torch.cuda.is_available() and next(self.encoder_model.parameters()).is_cuda:
                audio_tensor = audio_tensor.cuda()
            
            # Get embedding
            with torch.no_grad():
                embedding = self.encoder_model.encode_batch(audio_tensor)
            
            return embedding.squeeze(0)
            
        except Exception as e:
            print(f"Error extracting embedding from audio data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Add method to transcribe audio directly
    def transcribe_audio(self, audio_data):
        """
        Transcribe speech from audio data directly.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            str: Transcribed text or None if failed
        """
        if not hasattr(self, "recognizer"):
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
        
        try:
            # Convert to 16-bit PCM WAV format
            from io import BytesIO
            import wave
            
            # Convert float32 [-1, 1] to int16
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Write to in-memory file
            byte_io = BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Reset position
            byte_io.seek(0)
            
            # Use speech_recognition
            import speech_recognition as sr
            with sr.AudioFile(byte_io) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
                
        except Exception as e:
            print(f"Error transcribing audio data: {e}")
            return None
    
    # Add the methods to the VoiceRecognizer class
    VoiceRecognizer.verify_speaker_from_audio = verify_speaker_from_audio
    VoiceRecognizer.extract_embedding_from_audio = extract_embedding_from_audio
    VoiceRecognizer.transcribe_audio = transcribe_audio
    VoiceRecognizer.last_verification_score = 0.0


# Extend VoiceRecognizer when this module is imported
extend_voice_recognizer()


# Example usage when run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create speaker monitor
    monitor = ContinuousSpeakerMonitor()
    
    # Setup visualization
    plt.figure(figsize=(10, 6))
    plt.ion()  # Interactive mode
    
    # Create verification status display
    status_text = plt.text(0.5, 0.5, "No speaker verified", 
                          ha='center', va='center', fontsize=24,
                          transform=plt.gca().transAxes)
    
    # Update function
    def update_status():
        speaker = monitor.get_current_speaker()
        if speaker:
            status_text.set_text(f"Speaker: {speaker}\nConfidence: {monitor.verification_result.get_confidence():.2f}")
            status_text.set_color('green')
        else:
            status_text.set_text("No speaker verified")
            status_text.set_color('red')
        plt.draw()
        plt.pause(0.01)
    
    # Speaker change callback
    def on_speaker_change(change_info):
        print(f"\nSpeaker change: {change_info['previous_speaker']} -> {change_info['current_speaker']}")
        update_status()
    
    # Command callback
    def on_command(command_info):
        print(f"\nCommand from {command_info['speaker']}: {command_info['transcript']}")
    
    # Register callbacks
    monitor.add_speaker_change_callback(on_speaker_change)
    monitor.add_speech_command_callback(on_command)
    
    try:
        # Start monitoring
        monitor.start()
        print("Speaker monitoring started. Speak to test verification.")
        print("Press Ctrl+C to stop.")
        
        # Initial status update
        update_status()
        
        # Keep the main thread alive and updating the display
        while True:
            time.sleep(0.1)
            update_status()
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.stop()
        plt.close() 