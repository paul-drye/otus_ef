# Face & Voice Recognition for Robot Interaction

A Python-based authentication system that combines face and voice recognition technologies to identify authorized users.

## Project Overview

This system uses face recognition and voice verification to authenticate users before allowing interaction. The application is designed for an autonomous mobile robot equipped with a camera, microphone, and speaker.

### Features

- **Face Recognition**: Detects and recognizes faces using the camera.
- **Voice Recognition**: Verifies the speaker's identity using voice samples.
- **Combined Authentication**: Requires both face and voice to match for successful authentication.
- **Text-to-Speech Feedback**: Provides verbal feedback using a text-to-speech engine.
- **Interactive Mode**: Allows voice command input after successful authentication.

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera (webcam)
- Microphone
- Speaker

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/face-voice-auth.git
   cd face-voice-auth
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: The `dlib` library (required by `face_recognition`) may require additional build dependencies. See [dlib installation instructions](https://github.com/davisking/dlib#installation) for details.

3. Speaker verification model:
   This project uses pyannote.audio for speaker verification. If you need to use it with a Hugging Face token, modify the `voice_recognition_module.py` file.

## Usage

The system provides several command-line interfaces:

### User Enrollment

Enroll a new user with both face and voice samples:

```
python -m src.main enroll --name "John Doe"
```

Optional parameters:
- `--face-samples`: Number of face samples to capture (default: 5)
- `--voice-samples`: Number of voice samples to record (default: 3)

### Authentication

Authenticate a user:

```
python -m src.main auth
```

Optional parameters:
- `--face-time`: Time in seconds to attempt face recognition (default: 5)
- `--display-video`: Display video feed during face recognition

### Interactive Mode

Start an interactive session after successful authentication:

```
python -m src.main interactive
```

In this mode, you can speak commands and the system will respond accordingly.

### List Users

List all enrolled users:

```
python -m src.main list-users
```

## System Architecture

The project is organized into the following modules:

- **Face Recognition Module**: Handles face detection and recognition using the camera.
- **Voice Recognition Module**: Performs speech recognition and speaker verification.
- **Decision Engine**: Coordinates face and voice recognition for authentication.
- **Robot Interaction**: Provides verbal feedback through text-to-speech.

## Customization

You can customize various parameters by editing the `config/settings.yaml` file:

- Camera settings
- Face recognition tolerance
- Voice recognition parameters
- Text-to-speech properties

## Troubleshooting

### Common Issues

1. **Camera not working**: Ensure the camera is properly connected and not in use by another application. You can change the camera index in the configuration file.

2. **Microphone not detecting**: Check your microphone settings and permissions. Test it with another application to verify it's working.

3. **Face recognition issues**: Ensure adequate lighting for better face detection. You may need to adjust the tolerance parameter in the configuration.

4. **Voice recognition issues**: Speak clearly and in a quiet environment. You may need to adjust the verification threshold in the configuration.

## Future Enhancements

- Add face tracking capabilities
- Implement more sophisticated voice commands
- Add support for multiple languages
- Optimize for embedded devices

## License

This project is licensed under the MIT License - see the LICENSE file for details.
