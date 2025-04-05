import numpy as np
import sounddevice as sd
import time
from real_time_audio_processor import RealTimeAudioProcessor

def on_speech_detected(audio_segment: np.ndarray):
    """Callback for when speech is detected."""
    print(f"Speech detected! Length: {len(audio_segment) / 16000:.2f} seconds")
    
    # Play back the detected speech (optional)
    sd.play(audio_segment, 16000)
    sd.wait()

def on_transcription(text: str):
    """Callback for when transcription is available."""
    print(f"Transcription: {text}")

def main():
    # Initialize the audio processor
    processor = RealTimeAudioProcessor(
        sample_rate=16000,
        chunk_size=480,  # 30ms chunks
        vad_aggressiveness=3,
        min_speech_duration=0.5,
        max_speech_duration=2.0,
        silence_duration=0.5
    )
    
    # Set callbacks
    processor.set_callbacks(
        on_speech_detected=on_speech_detected,
        on_transcription=on_transcription
    )
    
    try:
        print("Starting real-time audio processing...")
        print("Press Ctrl+C to stop")
        
        # Start processing
        processor.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main() 