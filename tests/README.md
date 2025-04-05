# Testing Tools

This directory contains testing utilities for the face and voice recognition system.

## Available Tests

### Camera Tests
Test basic camera functionality and scan for available cameras:

```bash
# Scan for available cameras
python -m tests.test_camera --scan

# Test a specific camera
python -m tests.test_camera --camera 0

# Test camera and update the config file
python -m tests.test_camera --camera 1 --update-config
```

### Face Recognition Tests
Test various aspects of the face recognition module:

```bash
# Test face detection
python -m tests.test_face_recognition detection

# Test user enrollment
python -m tests.test_face_recognition enrollment --name "Your Name" --samples 3

# Test face recognition
python -m tests.test_face_recognition recognition
```

### Audio Tests
Test basic audio functionality and list available microphones:

```bash
# List available audio devices
python -m tests.test_audio list

# Test microphone recording
python -m tests.test_audio record --device 0 --duration 5

# Play back a recorded audio file
python -m tests.test_audio play --file test_recording.wav

# Test and update configuration
python -m tests.test_audio record --device 0 --update-config
```

### SpeechBrain Speaker Verification Tests (Current Implementation)
Test various aspects of the SpeechBrain-based voice recognition module:

```bash
# Test if the speaker verification model loads correctly
python -m tests.test_speechbrain_verification model

# Test embedding extraction
python -m tests.test_speechbrain_verification embedding

# Test user enrollment
python -m tests.test_speechbrain_verification enroll --name "Your Name"

# Test speaker verification
python -m tests.test_speechbrain_verification verify

# Test with automatic enrollment
python -m tests.test_speechbrain_verification verify --enroll-first
```

### ⚠️ Deprecated: Voice Recognition Tests (Old Implementation)
These tests use the old pyannote.audio implementation and are kept for reference only:

```bash
# Test audio recording functionality
python -m tests.test_voice_recognition record --duration 5

# Test speech-to-text conversion
python -m tests.test_voice_recognition transcribe

# Test if the speaker verification model loads correctly
python -m tests.test_voice_recognition model
```

### Integration Tests (Full System)
Test the complete face and voice authentication system:

```bash
# Test full enrollment and verification flow
python -m tests.test_integration full-flow --name "Test User"

# Test face verification only
python -m tests.test_integration face

# Test voice verification only
python -m tests.test_integration voice
```

## Test vs. Implementation Differences

### Camera Testing
There are two different test_camera implementations:

1. **tests/test_camera.py**:
   - Standalone utility for testing basic camera functionality
   - Can scan for available cameras on the system
   - Can update the configuration file with the correct camera index
   - Focuses on isolating hardware issues
   - Always releases the camera resource when done

2. **FaceRecognizer.test_camera() method**:
   - Part of the FaceRecognizer class in src/face_recognition_module.py
   - Uses the class's configured camera settings
   - Tests both camera functionality AND face detection
   - Designed to work within the context of the face recognition system
   - Doesn't release the camera (as other class methods might use it later)

### Audio Testing
Similarly, there are two approaches for audio testing:

1. **tests/test_audio.py**:
   - Standalone utility for testing basic microphone functionality
   - Lists all available audio devices
   - Provides visual feedback during recording (volume meter)
   - Generates waveform visualizations to check audio quality
   - Tests playback functionality
   - Updates configuration with correct device index

2. **VoiceRecognizer module tests**:
   - Tests the integrated audio functionality within the VoiceRecognizer class
   - Tests speech-to-text conversion
   - Tests speaker verification model loading
   - Tests user enrollment and verification

### Integration Testing
The integration tests combine multiple components to test the complete authentication flow:

1. **tests/test_integration.py**:
   - Tests the full enrollment and verification flow using the DecisionEngine
   - Tests face-only and voice-only verification separately
   - Allows testing with specific user names
   - Provides detailed feedback about each step of the process
   - Useful for end-to-end testing of the entire system

## Debugging Tips

### Camera Issues
- If no camera is detected, check if it's connected properly
- Try scanning for devices with `python -m tests.test_camera --scan`
- Test with another application like Cheese to verify it works
- If using a built-in laptop webcam, make sure it's not disabled in BIOS

### Audio Issues
- If no microphone is detected, check if it's connected properly
- List available devices with `python -m tests.test_audio list`
- Check system volume settings and ensure the microphone isn't muted
- Test with another application like Audacity to verify it works

### SpeechBrain Issues
- If you get CUDA/GPU errors, try setting `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32`
- For memory issues, decrease batch size or use CPU-only mode
- If model downloads fail, check your internet connection or download manually
- Models are stored in the `models/` directory for offline use 