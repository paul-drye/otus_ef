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
        
        # Speaker embeddings and user names
        self.speaker_embeddings = {}
        
        # Initialize speaker verification model from SpeechBrain
        try:
            # Import here to avoid loading the model if not needed
            from speechbrain.inference.speaker import EncoderClassifier
            
            print("Loading SpeechBrain speaker verification model...")
            self.verification_model = EncoderClassifier.from_hparams(
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
    
    def record_audio(self, output_path=None, duration=None):
        """
        Record audio from the microphone.
        
        Args:
            output_path (str, optional): Path to save the recorded audio
            duration (int, optional): Duration in seconds to record
            
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
        
        print(f"Recording audio for {duration} seconds...")
        
        # Open stream
        try:
            stream = p.open(format=audio_format,
                            channels=channels,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=self.audio_device_index,
                            frames_per_buffer=chunk)
            
            # Record audio with a progress bar
            frames = []
            for i in tqdm(range(0, int(self.sample_rate / chunk * duration)), 
                          desc="Recording", unit="chunk"):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            print("Recording finished.")
            
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
            audio_path = self.record_audio()
        
        print("Transcribing speech...")
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                print(f"Transcription: {text}")
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
            print(f"Extracting embedding from {audio_path}")
            # Load audio using speechbrain's method or convert your audio file
            signal = self.verification_model.load_audio(audio_path)
            
            # Get embedding (the model handles any necessary preprocessing)
            embedding = self.verification_model.encode_batch(signal.unsqueeze(0))
            
            print(f"Embedding extracted successfully, shape: {embedding.shape}")
            return embedding.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def enroll_user(self, user_name, num_samples=1, delay=1):
        """
        Enroll a new user by recording voice samples.
        
        Args:
            user_name (str): Name of the user to enroll
            num_samples (int): Number of voice samples to record (default is 1)
            delay (int): Delay between recordings in seconds
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        if not self.model_loaded:
            print("Speaker verification model not loaded. Cannot enroll user.")
            return False
        
        print(f"Enrolling user: {user_name}")
        print("Please speak clearly for voice recognition.")
        
        # Record audio sample
        audio_path = self.record_audio(duration=15)
        
        # Extract speaker embedding
        embedding = self.extract_speaker_embedding(audio_path)
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        if embedding is not None:
            # Store the embedding directly
            self.speaker_embeddings[user_name] = embedding
            
            # Save updated embeddings
            self._save_speaker_embeddings()
            
            print(f"User {user_name} enrolled successfully")
            return True
        else:
            print(f"Failed to enroll user {user_name}. Could not process voice sample.")
            return False
    
    def verify_speaker(self, audio_path=None):
        """
        Verify if the speaker matches any enrolled user using SpeechBrain.
        
        Args:
            audio_path (str, optional): Path to the audio file
            
        Returns:
            tuple: (verified, user_name) - whether a speaker was verified and the user's name
        """
        if not self.model_loaded:
            print("Speaker verification model not loaded.")
            return False, None
        
        if not self.speaker_embeddings:
            print("No enrolled speakers. Please enroll a user first.")
            return False, None
        
        if audio_path is None:
            audio_path = self.record_audio()
        
        # Extract speaker embedding
        embedding = self.extract_speaker_embedding(audio_path)
        
        if embedding is None:
            return False, None
        
        # Compare with enrolled speakers using cosine similarity
        best_match = None
        best_score = -float('inf')  # Initialize with lowest possible score
        
        for name, stored_embedding in self.speaker_embeddings.items():
            try:
                # Compute cosine similarity (higher is better)
                score = self._compute_similarity(embedding, stored_embedding)
                
                print(f"Similarity score for {name}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error comparing with user {name}: {e}")
                continue
        
        # Clean up temporary audio file
        if os.path.exists(audio_path) and audio_path.startswith(self.voice_data_path):
            os.remove(audio_path)
        
        # Verify if the best match is above the threshold
        # Note: For cosine similarity, higher is better (opposite of distance)
        if best_match and best_score > self.verification_threshold:
            print(f"Speaker verified as {best_match} (score: {best_score:.4f})")
            return True, best_match
        else:
            print(f"Speaker not verified (best score: {best_score:.4f}, threshold: {self.verification_threshold})")
            return False, None
    
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
        
        # Normalize embeddings (required for cosine similarity)
        embedding1 = embedding1 / torch.norm(embedding1)
        embedding2 = embedding2 / torch.norm(embedding2)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity = torch.dot(embedding1.flatten(), embedding2.flatten()).item()
        
        return similarity


if __name__ == "__main__":
    # Simple test
    recognizer = VoiceRecognizer()
    
    # Check if there are already enrolled users
    if not recognizer.speaker_embeddings:
        print("No enrolled users found. Starting enrollment...")
        recognizer.enroll_user("Test User")
    
    print("Starting speaker verification...")
    verified, name = recognizer.verify_speaker()
    
    if verified:
        print(f"Verified: {name}")
    else:
        print("Speaker not verified")
        
    # Test transcription
    text = recognizer.transcribe_speech()
    if text:
        print(f"You said: {text}")
