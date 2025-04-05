#!/usr/bin/env python3
import pyaudio
import wave
import argparse
import time
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def list_audio_devices():
    """
    List all available audio input devices.
    """
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print(f"Found {num_devices} audio devices:")
    
    input_devices = []
    
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info.get('name')
        max_input_channels = device_info.get('maxInputChannels')
        
        if max_input_channels > 0:
            input_devices.append(i)
            print(f"  Input Device {i}: {device_name}")
            print(f"    Max Input Channels: {max_input_channels}")
            print(f"    Default Sample Rate: {device_info.get('defaultSampleRate')}")
    
    p.terminate()
    return input_devices


def test_microphone(device_index=None, duration=5, output_path="test_recording.wav"):
    """
    Test microphone by recording audio and saving it to a file.
    
    Args:
        device_index (int, optional): Index of the microphone device to use
        duration (int): Duration in seconds to record
        output_path (str): Path to save the recorded audio
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Testing microphone{''+(' (device '+str(device_index)+')') if device_index is not None else ''}...")
    
    # Audio recording parameters
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    
    # Create PyAudio instance
    p = pyaudio.PyAudio()
    
    try:
        # Open stream
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)
        
        print(f"Recording audio for {duration} seconds...")
        print("Please speak into the microphone...")
        
        frames = []
        audio_data = []  # For plotting waveform
        
        # Record audio for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            
            # Convert data to numpy array for visualization
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_data.extend(audio_chunk)
            
            # Print a simple volume indicator
            volume = np.abs(audio_chunk).mean()
            bars = int(50 * volume / 32768)
            print(f"\rVolume: {'|' * bars}{' ' * (50 - bars)} {volume:.0f}", end='')
        
        print("\nRecording finished.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Analyze recording
        audio_array = np.array(audio_data)
        max_amplitude = np.abs(audio_array).max()
        mean_amplitude = np.abs(audio_array).mean()
        
        print(f"Audio statistics:")
        print(f"  Max amplitude: {max_amplitude} / 32768 ({max_amplitude/32768*100:.1f}%)")
        print(f"  Mean amplitude: {mean_amplitude:.1f} / 32768 ({mean_amplitude/32768*100:.1f}%)")
        
        # Plot waveform if not silent
        if max_amplitude > 1000:  # Threshold to determine if there's actual audio
            plt.figure(figsize=(10, 4))
            plt.plot(audio_array)
            plt.title('Audio Waveform')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            waveform_path = os.path.splitext(output_path)[0] + "_waveform.png"
            plt.savefig(waveform_path)
            print(f"Waveform saved to {waveform_path}")
            plt.close()
        
        # Save the recording to a WAV file
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"Audio saved to {output_path}")
        
        # Check if recording was too quiet
        if max_amplitude < 1000:
            print("\nWARNING: The recording is very quiet. Please check:")
            print("  - Your microphone is properly connected")
            print("  - Microphone volume settings in your OS")
            print("  - You're speaking close enough to the microphone")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error testing microphone: {e}")
        return False
    
    finally:
        p.terminate()


def play_audio(audio_path="test_recording.wav"):
    """
    Play back a recorded audio file.
    
    Args:
        audio_path (str): Path to the audio file to play
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found")
        return False
    
    print(f"Playing audio file: {audio_path}")
    
    # Get audio file info
    with wave.open(audio_path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Print audio info
        duration = n_frames / frame_rate
        print(f"Audio info:")
        print(f"  Channels: {channels}")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Frame rate: {frame_rate} Hz")
        print(f"  Frames: {n_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        
        # Create PyAudio instance
        p = pyaudio.PyAudio()
        
        try:
            # Open stream for playback
            stream = p.open(format=p.get_format_from_width(sample_width),
                            channels=channels,
                            rate=frame_rate,
                            output=True)
            
            # Read and play the audio file
            wf.rewind()
            data = wf.readframes(1024)
            
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            print("Playback finished")
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
            
        finally:
            p.terminate()


def update_audio_config(device_index=None):
    """
    Update the audio device index in the config file.
    
    Args:
        device_index (int, optional): Index of the audio device to use
    """
    import yaml
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/settings.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Add audio device settings if needed
        if 'audio_device' not in config:
            config['audio_device'] = {}
        
        if device_index is not None:
            config['audio_device']['input_device_index'] = device_index
            
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Updated audio device index to {device_index} in {config_path}")
        
    except Exception as e:
        print(f"Error updating config file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test audio recording and playback')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List devices command
    subparsers.add_parser('list', help='List available audio devices')
    
    # Test recording command
    record_parser = subparsers.add_parser('record', help='Test microphone recording')
    record_parser.add_argument('--device', type=int, help='Audio device index')
    record_parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds')
    record_parser.add_argument('--output', type=str, default='test_recording.wav', help='Output file path')
    record_parser.add_argument('--update-config', action='store_true', help='Update config file with device index')
    
    # Play audio command
    play_parser = subparsers.add_parser('play', help='Play recorded audio')
    play_parser.add_argument('--file', type=str, default='test_recording.wav', help='Audio file to play')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_audio_devices()
    
    elif args.command == 'record':
        success = test_microphone(device_index=args.device, 
                                 duration=args.duration,
                                 output_path=args.output)
        
        if success and args.update_config:
            update_audio_config(device_index=args.device)
    
    elif args.command == 'play':
        play_audio(audio_path=args.file)
    
    else:
        # Default behavior
        print("Audio Testing Utility")
        print("\nAvailable commands:")
        print("  python -m tests.test_audio list")
        print("  python -m tests.test_audio record [--device ID] [--duration SECONDS] [--update-config]")
        print("  python -m tests.test_audio play [--file PATH]")
        parser.print_help()


if __name__ == "__main__":
    main() 