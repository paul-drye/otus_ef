import numpy as np
import pyaudio
import webrtcvad
import queue
import threading
import time
from faster_whisper import WhisperModel
from typing import Optional, Callable, Tuple
import logging

class RealTimeAudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 480,  # 30ms at 16kHz
        vad_aggressiveness: int = 3,
        min_speech_duration: float = 0.5,
        max_speech_duration: float = 2.0,
        silence_duration: float = 0.5,
        device_index: Optional[int] = None
    ):
        """
        Initialize the real-time audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per audio chunk
            vad_aggressiveness: VAD aggressiveness (0-3)
            min_speech_duration: Minimum speech duration in seconds
            max_speech_duration: Maximum speech duration in seconds
            silence_duration: Duration of silence to consider speech segment complete
            device_index: Audio input device index
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.silence_duration = silence_duration
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Initialize audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            input_device_index=device_index
        )
        
        # Initialize queues and buffers
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.silence_counter = 0
        
        # Processing flags
        self.is_running = False
        self.processing_thread = None
        
        # Initialize Whisper model for transcription
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # Callbacks
        self.on_speech_detected: Optional[Callable[[np.ndarray], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the audio processing loop."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()
        self.logger.info("Started audio processing")
    
    def stop(self):
        """Stop the audio processing loop."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.logger.info("Stopped audio processing")
    
    def _process_audio(self):
        """Main audio processing loop."""
        while self.is_running:
            try:
                # Read audio chunk
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Check if chunk contains speech
                is_speech = self.vad.is_speech(audio_data, self.sample_rate)
                
                if is_speech:
                    self.speech_buffer.append(audio_array)
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1
                
                # Process speech buffer when silence is detected or max duration reached
                if (self.silence_counter >= int(self.silence_duration * self.sample_rate / self.chunk_size) or
                    len(self.speech_buffer) * self.chunk_size / self.sample_rate >= self.max_speech_duration):
                    
                    if len(self.speech_buffer) > 0:
                        # Concatenate speech segments
                        speech_segment = np.concatenate(self.speech_buffer)
                        
                        # Check if speech segment meets minimum duration
                        if len(speech_segment) / self.sample_rate >= self.min_speech_duration:
                            # Process speech segment
                            self._process_speech_segment(speech_segment)
                        
                        # Clear buffer
                        self.speech_buffer = []
                        self.silence_counter = 0
                
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
    
    def _process_speech_segment(self, speech_segment: np.ndarray):
        """Process a detected speech segment."""
        try:
            # Convert to float32 for Whisper
            audio_float32 = speech_segment.astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            segments, _ = self.whisper_model.transcribe(
                audio_float32,
                beam_size=5,
                language="en"
            )
            
            # Get transcription
            transcription = " ".join([segment.text for segment in segments])
            
            # Call callbacks
            if self.on_speech_detected:
                self.on_speech_detected(speech_segment)
            
            if self.on_transcription and transcription.strip():
                self.on_transcription(transcription)
                
        except Exception as e:
            self.logger.error(f"Error processing speech segment: {e}")
    
    def set_callbacks(
        self,
        on_speech_detected: Optional[Callable[[np.ndarray], None]] = None,
        on_transcription: Optional[Callable[[str], None]] = None
    ):
        """Set callback functions for speech detection and transcription."""
        self.on_speech_detected = on_speech_detected
        self.on_transcription = on_transcription 