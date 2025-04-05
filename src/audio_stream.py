import numpy as np
import pyaudio
import threading
import time
import queue
from typing import Optional, Callable, List, Tuple

class AudioStream:
    """
    Handles continuous audio streaming from microphone with buffer management.
    Provides callbacks for real-time audio processing.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 chunk_size: int = 1024,
                 device_index: Optional[int] = None):
        """
        Initialize the audio stream.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            device_index: Specific audio device to use, or None for default
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        # PyAudio setup
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Threading setup
        self.running = False
        self.audio_thread = None
        
        # Callback setup
        self.callbacks = []
        
        # Buffer for raw audio data
        self.buffer_queue = queue.Queue(maxsize=100)  # Limit buffer size
    
    def register_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register a callback function to be called with each audio chunk.
        
        Args:
            callback: Function that takes a numpy array of audio data
        """
        self.callbacks.append(callback)
    
    def start(self) -> None:
        """Start the audio stream and processing thread."""
        if self.running:
            print("Audio stream is already running.")
            return
        
        try:
            # Open PyAudio stream
            self.stream = self.pa.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start processing thread
            self.running = True
            self.audio_thread = threading.Thread(target=self._process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            print("Audio stream started.")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.stop()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function that receives raw audio data."""
        try:
            self.buffer_queue.put(in_data, block=False)
        except queue.Full:
            print("Warning: Audio buffer is full. Processing might be too slow.")
        return (None, pyaudio.paContinue)
    
    def _process_audio(self) -> None:
        """Process audio chunks from the buffer queue and call registered callbacks."""
        while self.running:
            try:
                # Get audio data from queue with timeout
                audio_data = self.buffer_queue.get(timeout=0.5)
                
                # Convert to numpy array (16-bit PCM to float32 normalized to [-1, 1])
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Call all registered callbacks with the numpy array
                for callback in self.callbacks:
                    try:
                        callback(audio_array)
                    except Exception as e:
                        print(f"Error in audio callback: {e}")
                
                # Mark task as done
                self.buffer_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, just continue
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                if not self.running:
                    break
    
    def stop(self) -> None:
        """Stop the audio stream and processing thread."""
        self.running = False
        
        # Stop and close the PyAudio stream
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.stream = None
        
        # Clear the buffer queue
        while not self.buffer_queue.empty():
            try:
                self.buffer_queue.get_nowait()
                self.buffer_queue.task_done()
            except:
                break
        
        # Wait for processing thread to finish
        if self.audio_thread is not None and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
            self.audio_thread = None
        
        print("Audio stream stopped.")
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        self.stop()
        
        if self.pa is not None:
            self.pa.terminate()
            self.pa = None


# Example usage when run directly
if __name__ == "__main__":
    # Simple callback to print audio levels
    def print_audio_level(audio_data):
        level = np.abs(audio_data).mean()
        bars = int(50 * level)
        print(f"\rAudio level: {'|' * bars}{' ' * (50 - bars)} {level:.4f}", end='')
    
    # Create and start audio stream
    audio_stream = AudioStream()
    audio_stream.register_callback(print_audio_level)
    
    try:
        audio_stream.start()
        print("Audio streaming started. Press Ctrl+C to stop.")
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        audio_stream.stop() 