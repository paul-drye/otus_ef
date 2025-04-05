import os
import yaml
import numpy as np
import pickle
import speech_recognition as sr
import torch
import time
import wave
import pyaudio
from tqdm import tqdm


class VoiceRecognizer:
    def __init__(self, config_path='config/settings.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.voice_config = self.config['voice_recognition']
        self.voice_data_path = self.config['paths']['voice_data']
        self.model_path = self.config['paths']['models']
        
        # Ensure the voice data directory exists
        os.makedirs(self.voice_data_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        # Parameters
        self.record_duration = self.voice_config['record_duration']
        self.sample_rate = self.voice_config['sample_rate']
        self.verification_threshold = self.voice_config.get('verification_threshold', 0.25)
        
        # Get audio device index from config if available
        self.audio_device_index = None
        if 'audio_device' in self.config and 'input_device_index' in self.config['audio_device']:
            self.audio_device_index = self.config['audio_device']['input_device_index']
        
        # Speech recognition for transcription
        self.recognizer = sr.Recognizer()
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        # Speaker embeddings and user names
        self.speaker_embeddings = {}
        
        # Initialize speaker verification model from SpeechBrain
        try:
            # Import here to avoid loading the model if not needed
            from speechbrain.inference.speaker import EncoderClassifier
            
            print("Loading SpeechBrain speaker verification model...")
            self.encoder_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(self.model_path, "speechbrain_spkrec")
            )
            self.model_loaded = True
            print("Speaker verification model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load speaker verification model: {e}")
            print("Speaker verification will be disabled.")
            self.model_loaded = False
        
        # Load saved speaker embeddings if available
        self._load_speaker_embeddings()
    
    def _load_speaker_embeddings(self):
        """Load known speaker embeddings from the data directory."""
        embeddings_file = os.path.join(self.voice_data_path, 'speaker_embeddings.pkl')
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as file:
                    self.speaker_embeddings = pickle.load(file)
                print(f"Loaded speaker embeddings for {len(self.speaker_embeddings)} users")
            except Exception as e:
                print(f"Error loading speaker embeddings: {e}")
    
    def _save_speaker_embeddings(self):
        """Save speaker embeddings to the data directory."""
        embeddings_file = os.path.join(self.voice_data_path, 'speaker_embeddings.pkl')
        with open(embeddings_file, 'wb') as file:
            pickle.dump(self.speaker_embeddings, file)
    
    def record_audio(self, output_path=None, duration=None, countdown=False):
        """
        Record audio from the microphone.
        
        Args:
            output_path (str, optional): Path to save the recorded audio
            duration (int, optional): Duration in seconds to record
            countdown (bool): Whether to show a countdown before recording
            
        Returns:
            str: Path to the recorded audio file
        """
        if duration is None:
            duration = self.record_duration
            
        if output_path is None:
            output_path = os.path.join(self.voice_data_path, f"temp_recording_{int(time.time())}.wav")
        
        # Audio recording parameters
        chunk = 1024
        audio_format = pyaudio.paInt16
        channels = 1
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Display countdown if requested
        if countdown:
            print("\nGet ready to speak!")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            print("Recording NOW - please speak clearly...")
        
        # Open stream
        try:
            stream = p.open(format=audio_format,
                            channels=channels,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=self.audio_device_index,
                            frames_per_buffer=chunk)
            
            # Record audio
            frames = []
            for i in range(0, int(self.sample_rate / chunk * duration)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save the recorded audio to a WAV file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            return output_path
            
        except Exception as e:
            p.terminate()
            print(f"Error recording audio: {e}")
            raise
    
    def transcribe_speech(self, audio_path=None):
        """
        Convert speech to text.
        
        Args:
            audio_path (str, optional): Path to the audio file
            
        Returns:
            str: Transcribed text or None if failed
        """
        if audio_path is None:
            # Record audio without countdown for faster response
            audio_path = self.record_audio(countdown=False)
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                
                # Clean up temporary file if we created it
                if os.path.exists(audio_path) and audio_path.startswith(self.voice_data_path):
                    os.remove(audio_path)
                
                return text
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Speech Recognition service; {e}")
            return None
    
    def extract_speaker_embedding(self, audio_path):
        """
        Extract speaker embedding from audio file using SpeechBrain.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Speaker embedding tensor or None if failed
        """
        if not self.model_loaded:
            print("Speaker verification model not loaded.")
            return None
        
        try:
            # Load audio using speechbrain's method or convert your audio file
            signal = self.encoder_model.load_audio(audio_path)
            
            # Get embedding (the model handles any necessary preprocessing)
            embedding = self.encoder_model.encode_batch(signal.unsqueeze(0))
            
            return embedding.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def enroll_user(self, user_name, num_samples=1, delay=1, confirm=True):
        """
        Enroll a new user by recording voice samples.
        
        Args:
            user_name (str): Name of the user to enroll
            num_samples (int): Number of voice samples to record (default is 1)
            delay (int): Delay between recordings in seconds
            confirm (bool): Whether to ask for confirmation before saving
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        if not self.model_loaded:
            print("Speaker verification model not loaded. Cannot enroll user.")
            return False
        
        # Create enrollment debug directory
        debug_dir = os.path.join(self.voice_data_path, f"debug_{user_name}_{int(time.time())}")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Enrollment loop - continue until user is satisfied or cancels
        enrollment_complete = False
        
        while not enrollment_complete:
            print(f"\nEnrolling user: {user_name}")
            print("Please prepare to speak for about 15 seconds.")
            print("Speak continuously and clearly for best results.")
            print("\nSample phrases you can use:")
            print(" - 'The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly.'")
            print(" - 'My voice is my password. Please verify me with my voice print.'")
            print(" - Or describe your favorite place, hobby, or tell a short story.")
            time.sleep(3)  # Give the user time to read instructions
            
            # Record audio sample with countdown
            audio_path = self.record_audio(duration=15, countdown=True)
            
            # Save a copy for debugging/verification
            debug_audio_path = os.path.join(debug_dir, f"enrollment_{int(time.time())}.wav")
            import shutil
            shutil.copy2(audio_path, debug_audio_path)
            print(f"Saved enrollment audio to {debug_audio_path}")
            
            # Extract speaker embedding
            embedding = self.extract_speaker_embedding(audio_path)
            
            if embedding is None:
                print(f"Failed to extract voice embedding from the recording.")
                
                if confirm:
                    retry = input("\nFailed to process voice. Would you like to try again? (y/n): ").lower().strip()
                    if retry == 'y' or retry == 'yes':
                        # Clean up temporary audio file
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                        continue
                    else:
                        return False
                else:
                    # Clean up temporary audio file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    return False
            
            # Test verification with the same recording
            print("\nTesting voice quality with a verification test...")
            
            # Convert embedding to match storage format
            test_embedding = embedding
            
            # Compare with existing users (if any)
            potential_matches = []
            
            for name, stored_embedding in self.speaker_embeddings.items():
                if name != user_name:  # Skip if comparing to a previous version of the same user
                    # Compute similarity
                    score = self._compute_similarity(test_embedding, stored_embedding)
                    
                    # Check if too similar to an existing user
                    if score > self.verification_threshold:
                        potential_matches.append((name, score))
            
            # Sort potential matches by score (highest first)
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Warn if voice is too similar to existing users
            if potential_matches:
                print("\nWARNING: Your voice is similar to existing enrolled users:")
                for name, score in potential_matches:
                    print(f" - {name}: similarity score {score:.4f}")
                
                if confirm:
                    print("\nThis might cause confusion during verification.")
                    proceed = input("Would you like to proceed anyway or try again with a more distinctive voice? (p/t): ").lower().strip()
                    
                    if proceed == 't' or proceed == 'try':
                        # Clean up temporary audio file
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                        continue
            
            # Ask for confirmation if requested
            if confirm:
                # Transcribe the audio to let the user verify what was recorded
                try:
                    with sr.AudioFile(audio_path) as source:
                        audio_data = self.recognizer.record(source)
                        transcription = self.recognizer.recognize_google(audio_data)
                        print(f"\nTranscription of your enrollment audio: \"{transcription}\"")
                except Exception as e:
                    print(f"Could not transcribe audio: {e}")
                
                # Let the user hear their recording
                play_back = input("\nWould you like to listen to your recording? (y/n): ").lower().strip()
                if play_back == 'y' or play_back == 'yes':
                    try:
                        print("Playing back your recording...")
                        # Try to use platform-specific audio playback
                        import platform
                        system = platform.system()
                        
                        if system == 'Darwin':  # macOS
                            os.system(f"afplay {audio_path}")
                        elif system == 'Windows':
                            os.system(f"start {audio_path}")
                        else:  # Linux or other
                            try:
                                import sounddevice as sd
                                import soundfile as sf
                                data, fs = sf.read(audio_path)
                                sd.play(data, fs)
                                sd.wait()
                            except ImportError:
                                os.system(f"aplay {audio_path}")
                    except Exception as e:
                        print(f"Error playing audio: {e}")
                
                # Ask user to confirm enrollment
                confirmation = input(f"\nDo you want to save this voice enrollment for {user_name}? (y/n): ").lower().strip()
                
                if confirmation != 'y' and confirmation != 'yes':
                    retry = input("Would you like to try again? (y/n): ").lower().strip()
                    if retry == 'y' or retry == 'yes':
                        # Clean up temporary audio file
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                        continue
                    else:
                        return False
            
            # User confirmed or confirmation not required
            # Store the embedding
            self.speaker_embeddings[user_name] = embedding
            
            # Save updated embeddings
            self._save_speaker_embeddings()
            
            print(f"User {user_name} enrolled successfully")
            print(f"Debug audio saved to {debug_audio_path}")
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Set flag to exit the loop
            enrollment_complete = True
            return True
    
    def verify_speaker(self, audio_path=None):
        """
        Verify a speaker's identity from an audio file or recording.
        
        Args:
            audio_path (str, optional): Path to an audio file for verification
                                        If None, audio will be recorded
        
        Returns:
            Tuple[bool, str, float]: Verification result, user name if verified, and confidence score
        """
        if not self.model_loaded:
            print("Speaker verification model not loaded.")
            return False, None, 0.0
        
        if audio_path is None:
            # Record audio
            audio_path = self.record_audio(countdown=False)
        
        # Extract speaker embedding
        embedding = self.extract_speaker_embedding(audio_path)
        
        if embedding is None:
            return False, None, 0.0
        
        # If there are no enrolled speakers, verification fails
        if not self.speaker_embeddings:
            return False, None, 0.0
        
        # Compare with known speaker embeddings
        best_match = None
        best_score = -1.0
        
        for name, stored_embedding in self.speaker_embeddings.items():
            score = self._compute_similarity(embedding, stored_embedding)
            
            if score > best_score:
                best_score = score
                best_match = name
        
        # Verify if the score exceeds the threshold
        if best_score > self.verification_threshold:
            return True, best_match, best_score
        else:
            return False, None, best_score
    
    def _compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (torch.Tensor): First embedding
            embedding2 (torch.Tensor): Second embedding
            
        Returns:
            float: Cosine similarity score (higher means more similar)
        """        
        # Ensure embeddings are torch tensors with correct shape
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1)
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2)
        
        # Reshape tensors for compatibility with torch.nn.CosineSimilarity
        # CosineSimilarity expects [batch_size, vector_dim]
        embedding1 = embedding1.view(1, -1)
        embedding2 = embedding2.view(1, -1)
        
        # Use the torch.nn.CosineSimilarity instance
        similarity = self.similarity(embedding1, embedding2).item()
        
        return similarity


if __name__ == "__main__":
    # Simple test
    recognizer = VoiceRecognizer()
    
    # Check if there are already enrolled users
    if not recognizer.speaker_embeddings:
        print("No enrolled users found. Starting enrollment...")
        recognizer.enroll_user("Test User")
    
    print("Starting speaker verification...")
    verified, name, score = recognizer.verify_speaker()
    
    if verified:
        print(f"Verified: {name} with confidence score {score:.4f}")
    else:
        print("Speaker not verified")
        
    # Test transcription
    text = recognizer.transcribe_speech()
    if text:
        print(f"You said: {text}")
