import os
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable
import time
from collections import deque

# Check if SpeechBrain is installed, if not try to install it
from speechbrain.inference.VAD import VAD


class VoiceActivityDetector:
    """
    Voice Activity Detector using SpeechBrain.
    Detects speech in audio streams with configurable sensitivity.
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 sample_rate: int = 16000,
                 threshold: float = 0.5,
                 neg_threshold: float = 0.3,
                 activation_window: int = 8,
                 deactivation_window: int = 30):
        """
        Initialize the Voice Activity Detector.
        
        Args:
            model_dir: Directory to store/load the VAD model
            sample_rate: Audio sample rate in Hz
            threshold: Threshold for speech detection (0.0-1.0)
            neg_threshold: Threshold for speech absence (0.0-1.0)
            activation_window: Number of consecutive frames to trigger activation
            deactivation_window: Number of consecutive frames to trigger deactivation
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.activation_window = activation_window
        self.deactivation_window = deactivation_window
        
        # Model directory
        if model_dir is None:
            model_dir = os.path.join(os.path.expanduser("~"), ".cache", "vad_models")
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        
        # State for continuous processing
        self.speech_scores = deque(maxlen=max(activation_window, deactivation_window))
        self.is_speech_active = False
        self.last_prob = 0.0
        
        # Callbacks
        self.speech_start_callbacks = []
        self.speech_end_callbacks = []
        self.state_callbacks = []  # For tracking state changes
    
    def _load_model(self) -> None:
        """Load the SpeechBrain VAD model."""
        try:
            # Determine device
            if torch.cuda.is_available():
                print("Using GPU for VAD")
                device = "cuda"
            else:
                print("Using CPU for VAD")
                device = "cpu"
            
            # Load the model
            self.vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir=os.path.join(self.model_dir, "vad"),
                run_opts={"device": device}
            )
            
            print("VAD model loaded successfully")
        except Exception as e:
            print(f"Error loading VAD model: {e}")
            raise
    
    def add_speech_start_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to be called when speech starts.
        
        Args:
            callback: Function to be called when speech starts
        """
        self.speech_start_callbacks.append(callback)
    
    def add_speech_end_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to be called when speech ends.
        
        Args:
            callback: Function to be called when speech ends
        """
        self.speech_end_callbacks.append(callback)
    
    def register_state_callback(self, callback: Callable[[bool, float], None]) -> None:
        """
        Register a callback for tracking speech state changes for visualization.
        
        Args:
            callback: Function that takes (is_speech, probability) arguments
        """
        self.state_callbacks.append(callback)
    
    def process_audio(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Process audio data to detect speech.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (is_speech_detected, speech_probability)
        """
        try:
            # Ensure audio is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            
            # Move to the same device as the model
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Get speech probability
            speech_prob = self.vad_model.get_speech_prob_chunk(audio_tensor)[0].item()
            self.last_prob = speech_prob
            
            # Update window
            self.speech_scores.append(speech_prob)
            
            # Determine speech state
            prev_state = self.is_speech_active
            
            # Logic for state changes with hysteresis:
            # If currently inactive, we need several high scores to activate
            # If currently active, we need several low scores to deactivate
            if not self.is_speech_active:
                # Check for activation (need consistent high scores)
                if len(self.speech_scores) >= self.activation_window:
                    recent_scores = list(self.speech_scores)[-self.activation_window:]
                    if all(score >= self.threshold for score in recent_scores):
                        self.is_speech_active = True
                        self._trigger_speech_start()
            else:
                # Check for deactivation (need consistent low scores)
                if len(self.speech_scores) >= self.deactivation_window:
                    recent_scores = list(self.speech_scores)[-self.deactivation_window:]
                    if all(score <= self.neg_threshold for score in recent_scores):
                        self.is_speech_active = False
                        self._trigger_speech_end()
            
            # Notify state callbacks
            for callback in self.state_callbacks:
                try:
                    callback(self.is_speech_active, speech_prob)
                except Exception as e:
                    print(f"Error in state callback: {e}")
            
            return self.is_speech_active, speech_prob
            
        except Exception as e:
            print(f"Error processing audio for VAD: {e}")
            return False, 0.0
    
    def _trigger_speech_start(self) -> None:
        """Trigger callbacks for speech start event."""
        for callback in self.speech_start_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in speech start callback: {e}")
    
    def _trigger_speech_end(self) -> None:
        """Trigger callbacks for speech end event."""
        for callback in self.speech_end_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in speech end callback: {e}")
    
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """
        Simple check if the audio contains speech.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if speech is detected, False otherwise
        """
        is_speech, _ = self.process_audio(audio_data)
        return is_speech
    
    def get_speech_probability(self, audio_data: np.ndarray) -> float:
        """
        Get the probability of speech in the audio.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Probability of speech (0.0-1.0)
        """
        _, prob = self.process_audio(audio_data)
        return prob


# Example usage when run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import pyaudio
    
    # Parameters
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1600  # 100ms chunks
    
    # Initialize VAD
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
    
    # Initialize audio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                   channels=1,
                   rate=SAMPLE_RATE,
                   input=True,
                   frames_per_buffer=CHUNK_SIZE)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 100)
    ax.set_ylabel('Speech Probability')
    ax.set_xlabel('Time')
    line, = ax.plot([], [], lw=2)
    speech_indicator = ax.text(0.02, 0.95, 'SILENCE', transform=ax.transAxes, 
                              fontsize=24, color='red')
    
    # Data for visualization
    speech_probs = []
    speech_state = []
    
    def update_plot(frame):
        # Read audio
        audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.float32)
        
        # Process audio
        is_speech, prob = vad.process_audio(audio_data)
        
        # Update data
        speech_probs.append(prob)
        speech_state.append(is_speech)
        
        # Keep only the last 100 points
        if len(speech_probs) > 100:
            speech_probs.pop(0)
            speech_state.pop(0)
        
        # Update line
        line.set_data(range(len(speech_probs)), speech_probs)
        
        # Update speech indicator
        if is_speech:
            speech_indicator.set_text('SPEECH')
            speech_indicator.set_color('green')
        else:
            speech_indicator.set_text('SILENCE')
            speech_indicator.set_color('red')
        
        return line, speech_indicator
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_plot, frames=None, 
                                 interval=50, blit=True)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate() 