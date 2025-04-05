# Continuous Speaker Verification System

This system implements real-time continuous speaker verification using SpeechBrain. It constantly monitors audio input, verifies who is speaking, and can transcribe speech commands from verified speakers.

## Features

- **Continuous audio monitoring** - Streams audio from your microphone in real-time
- **Speaker verification** - Identifies who is speaking with confidence tracking
- **Voice activity detection** - Accurately detects when someone is speaking
- **Command processing** - Captures complete utterances and transcribes them
- **Visualization** - Real-time display of verification status and audio levels

## Components

The system consists of several modular components:

1. **AudioStream** (`src/audio_stream.py`) - Handles continuous audio streaming from the microphone
2. **BufferManager** (`src/buffer_manager.py`) - Manages audio buffers for verification and commands
3. **VoiceActivityDetector** (`src/voice_activity_detector.py`) - Detects when someone is speaking
4. **ContinuousSpeakerMonitor** (`src/continuous_speaker_monitor.py`) - Main component integrating all others
5. **Demo application** (`src/demo_app.py`) - Interactive visualization of the system

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Speaker Enrollment

Before using the system, you need to enroll speakers for verification:

```bash
python src/demo_app.py --enroll --name "John Doe"
```

Follow the prompts to record a voice sample. Repeat for each speaker you want to verify.

### Running the Demo

To start the continuous verification demo:

```bash
python src/demo_app.py
```

This will launch a visualization window showing:
- Current verified speaker and confidence
- Audio level display
- Speech activity indicator
- Command history

### Integration in Your Own Projects

To use the continuous speaker verification in your own projects:

```python
from src.continuous_speaker_monitor import ContinuousSpeakerMonitor

# Create the monitor
monitor = ContinuousSpeakerMonitor()

# Register callbacks
def on_speaker_change(change_info):
    print(f"Speaker changed: {change_info['previous_speaker']} -> {change_info['current_speaker']}")

def on_command(command_info):
    print(f"Command from {command_info['speaker']}: {command_info['transcript']}")

monitor.add_speaker_change_callback(on_speaker_change)
monitor.add_speech_command_callback(on_command)

# Start monitoring
monitor.start()

# ... your application logic ...

# Stop when done
monitor.stop()
```

## System Requirements

- Python 3.7+
- Microphone input device
- SpeechBrain (automatically installed with requirements)
- GPU recommended for better performance but not required

## Customization

The system is designed to be customizable:

- **Verification parameters** - Adjust window size, threshold, etc.
- **Voice activity detection** - Configure sensitivity
- **Audio parameters** - Change sample rate, device, etc.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
