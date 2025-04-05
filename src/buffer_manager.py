import numpy as np
import collections
from typing import Optional, List, Tuple, Dict, Any

class AudioBuffer:
    """Base class for audio buffers"""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def add(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer"""
        raise NotImplementedError("Subclasses must implement add()")
        
    def get(self) -> np.ndarray:
        """Get audio data from buffer"""
        raise NotImplementedError("Subclasses must implement get()")
        
    def clear(self) -> None:
        """Clear the buffer"""
        raise NotImplementedError("Subclasses must implement clear()")


class SlidingWindowBuffer(AudioBuffer):
    """
    Sliding window buffer for audio analysis.
    Maintains a fixed-duration buffer with the most recent audio.
    """
    
    def __init__(self, 
                 duration_seconds: float = 2.0, 
                 sample_rate: int = 16000,
                 step_size_seconds: Optional[float] = None):
        """
        Initialize sliding window buffer.
        
        Args:
            duration_seconds: Duration of audio to maintain in seconds
            sample_rate: Audio sample rate in Hz
            step_size_seconds: Step size for sliding window in seconds (None means same as chunk size)
        """
        super().__init__(sample_rate)
        self.duration_seconds = duration_seconds
        self.step_size_seconds = step_size_seconds
        
        # Calculate buffer size in samples
        self.buffer_size = int(duration_seconds * sample_rate)
        
        # Initialize empty buffer
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Position tracking
        self.is_full = False
        self.samples_added = 0
        
        # Last accessed window (for debugging/metrics)
        self.last_window_time = 0.0
    
    def add(self, audio_chunk: np.ndarray) -> bool:
        """
        Add audio chunk to the buffer, sliding the window.
        
        Args:
            audio_chunk: Numpy array of audio samples
            
        Returns:
            bool: True if buffer is full after adding (ready for processing)
        """
        chunk_size = len(audio_chunk)
        
        # If chunk is larger than buffer, just take the most recent portion
        if chunk_size >= self.buffer_size:
            self.buffer = audio_chunk[-self.buffer_size:]
            self.is_full = True
            self.samples_added += self.buffer_size
            return True
        
        # Shift buffer left by chunk size
        self.buffer = np.roll(self.buffer, -chunk_size)
        
        # Insert new chunk at the end
        self.buffer[-chunk_size:] = audio_chunk
        
        # Update position tracking
        self.samples_added += chunk_size
        if self.samples_added >= self.buffer_size:
            self.is_full = True
        
        return self.is_full
    
    def get(self) -> np.ndarray:
        """
        Get the current window of audio.
        
        Returns:
            numpy.ndarray: Current audio window
        """
        self.last_window_time = self.samples_added / self.sample_rate
        return self.buffer.copy()
    
    def get_duration(self) -> float:
        """Get the buffer duration in seconds"""
        return self.duration_seconds
    
    def get_if_step_completed(self) -> Optional[np.ndarray]:
        """
        Get window only if a full step has been completed since the last access.
        
        Returns:
            numpy.ndarray or None: Current window if step completed, otherwise None
        """
        if self.step_size_seconds is None:
            return self.get()
            
        step_size_samples = int(self.step_size_seconds * self.sample_rate)
        current_time = self.samples_added / self.sample_rate
        
        # Check if enough time has passed since last window
        if current_time - self.last_window_time >= self.step_size_seconds:
            return self.get()
        
        return None
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.is_full = False
        self.samples_added = 0
        self.last_window_time = 0.0


class DynamicBuffer(AudioBuffer):
    """
    Dynamic-sized buffer for capturing complete utterances.
    Grows as needed to contain a full speech segment.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 max_duration_seconds: float = 60.0,
                 initial_capacity_seconds: float = 5.0):
        """
        Initialize a dynamic buffer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            max_duration_seconds: Maximum buffer duration in seconds
            initial_capacity_seconds: Initial buffer capacity in seconds
        """
        super().__init__(sample_rate)
        
        self.max_size = int(max_duration_seconds * sample_rate)
        initial_size = int(initial_capacity_seconds * sample_rate)
        
        # Use a deque for efficient append operations
        self.buffer = collections.deque(maxlen=self.max_size)
        
        # Metadata
        self.duration = 0.0
        self.is_speech_active = False
        self.metadata = {
            "start_time": None,
            "end_time": None,
            "speech_segments": []
        }
    
    def add(self, audio_chunk: np.ndarray) -> float:
        """
        Add audio chunk to the buffer.
        
        Args:
            audio_chunk: Numpy array of audio samples
            
        Returns:
            float: Current buffer duration in seconds
        """
        # Add all samples to the buffer (deque handles overflow)
        for sample in audio_chunk:
            self.buffer.append(sample)
        
        # Update duration
        self.duration = len(self.buffer) / self.sample_rate
        
        return self.duration
    
    def get(self) -> np.ndarray:
        """
        Get the current buffer contents as a numpy array.
        
        Returns:
            numpy.ndarray: Current buffer contents
        """
        return np.array(self.buffer, dtype=np.float32)
    
    def get_with_metadata(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get buffer contents with metadata.
        
        Returns:
            tuple: (audio_data, metadata)
        """
        return self.get(), self.metadata.copy()
    
    def clear(self) -> None:
        """Clear the buffer and reset metadata"""
        self.buffer.clear()
        self.duration = 0.0
        self.is_speech_active = False
        self.metadata = {
            "start_time": None,
            "end_time": None,
            "speech_segments": []
        }
    
    def mark_speech_start(self, timestamp: Optional[float] = None) -> None:
        """
        Mark the start of speech in the buffer.
        
        Args:
            timestamp: Timestamp in seconds, or None for current buffer end
        """
        if timestamp is None:
            timestamp = self.duration
            
        if not self.is_speech_active:
            self.is_speech_active = True
            if self.metadata["start_time"] is None:
                self.metadata["start_time"] = timestamp
            
            # Start a new speech segment
            self.metadata["speech_segments"].append({"start": timestamp, "end": None})
    
    def mark_speech_end(self, timestamp: Optional[float] = None) -> None:
        """
        Mark the end of speech in the buffer.
        
        Args:
            timestamp: Timestamp in seconds, or None for current buffer end
        """
        if timestamp is None:
            timestamp = self.duration
            
        if self.is_speech_active:
            self.is_speech_active = False
            self.metadata["end_time"] = timestamp
            
            # Complete the current speech segment if there is one
            if self.metadata["speech_segments"] and self.metadata["speech_segments"][-1]["end"] is None:
                self.metadata["speech_segments"][-1]["end"] = timestamp
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds"""
        return self.duration


class BufferManager:
    """
    Manages different types of audio buffers.
    Coordinates between verification windows and command buffers.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 verification_window_seconds: float = 2.0,
                 verification_step_seconds: float = 0.5):
        """
        Initialize the buffer manager.
        
        Args:
            sample_rate: Audio sample rate in Hz
            verification_window_seconds: Size of verification window in seconds
            verification_step_seconds: Step size for verification window in seconds
        """
        self.sample_rate = sample_rate
        
        # Create buffers
        self.verification_buffer = SlidingWindowBuffer(
            duration_seconds=verification_window_seconds,
            sample_rate=sample_rate,
            step_size_seconds=verification_step_seconds
        )
        
        self.command_buffer = DynamicBuffer(
            sample_rate=sample_rate
        )
        
        # State tracking
        self.is_command_active = False
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Process an audio chunk through all buffers.
        
        Args:
            audio_chunk: Numpy array of audio samples
            
        Returns:
            dict: Status information about the buffers
        """
        # Add to verification buffer
        verification_ready = self.verification_buffer.add(audio_chunk)
        
        # Add to command buffer if active
        if self.is_command_active:
            self.command_buffer.add(audio_chunk)
        
        # Prepare status information
        status = {
            "verification_ready": verification_ready,
            "verification_window": self.verification_buffer.get_if_step_completed() if verification_ready else None,
            "command_active": self.is_command_active,
            "command_duration": self.command_buffer.get_duration() if self.is_command_active else 0.0
        }
        
        return status
    
    def start_command(self) -> None:
        """Start capturing a command"""
        if not self.is_command_active:
            self.is_command_active = True
            self.command_buffer.clear()
            self.command_buffer.mark_speech_start()
    
    def end_command(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        End command capture and return the buffer.
        
        Returns:
            tuple: (audio_data, metadata)
        """
        if self.is_command_active:
            self.command_buffer.mark_speech_end()
            self.is_command_active = False
            return self.command_buffer.get_with_metadata()
        
        return np.array([], dtype=np.float32), {}
    
    def get_verification_window(self) -> np.ndarray:
        """Get the current verification window"""
        return self.verification_buffer.get()
    
    def get_command_buffer(self) -> np.ndarray:
        """Get the current command buffer"""
        return self.command_buffer.get()
    
    def clear_all(self) -> None:
        """Clear all buffers"""
        self.verification_buffer.clear()
        self.command_buffer.clear()
        self.is_command_active = False


# Example usage when run directly
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from src.audio_stream import AudioStream
    
    # Create buffer manager
    buffer_manager = BufferManager(verification_window_seconds=2.0)
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    plt.ion()  # Interactive mode
    
    # Verification window display
    ax1 = plt.subplot(211)
    verification_line, = ax1.plot([], [])
    ax1.set_title("Verification Window (2s)")
    ax1.set_ylim(-1, 1)
    
    # Command buffer display
    ax2 = plt.subplot(212)
    command_line, = ax2.plot([], [])
    ax2.set_title("Command Buffer (Variable Length)")
    ax2.set_ylim(-1, 1)
    
    # Update function for visualization
    def update_plot():
        # Update verification window
        v_data = buffer_manager.get_verification_window()
        verification_line.set_data(np.arange(len(v_data)), v_data)
        ax1.set_xlim(0, len(v_data))
        
        # Update command buffer
        c_data = buffer_manager.get_command_buffer()
        command_line.set_data(np.arange(len(c_data)), c_data)
        ax2.set_xlim(0, max(len(c_data), 1))
        ax2.set_title(f"Command Buffer ({len(c_data)/16000:.1f}s)")
        
        plt.draw()
        plt.pause(0.01)
    
    # Audio processing callback
    command_active = False
    def process_audio(audio_data):
        global command_active
        
        # Process through buffer manager
        status = buffer_manager.process_audio_chunk(audio_data)
        
        # Check audio level for demonstration purposes
        level = np.abs(audio_data).mean()
        
        # Simulate VAD with simple threshold
        if level > 0.01 and not command_active:
            print("Speech detected, starting command capture")
            buffer_manager.start_command()
            command_active = True
        elif level < 0.005 and command_active:
            print("Speech ended, finishing command capture")
            audio, metadata = buffer_manager.end_command()
            command_active = False
            print(f"Captured {len(audio)/16000:.1f}s audio command")
        
        # Update visualization
        update_plot()
    
    # Create and start audio stream
    audio_stream = AudioStream(chunk_size=1024)
    audio_stream.register_callback(process_audio)
    
    try:
        audio_stream.start()
        print("Audio streaming started. Speak to see the buffers in action.")
        print("Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_stream.stop()
        plt.close() 