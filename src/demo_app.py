#!/usr/bin/env python3
"""
Demonstration application for the continuous speaker verification system.
Shows real-time speaker verification and command transcription.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from threading import Lock
from typing import Dict, List, Any, Optional

from continuous_speaker_monitor import ContinuousSpeakerMonitor
from voice_recognition_module import VoiceRecognizer

# Command history for display
class CommandHistory:
    """Stores and manages command history with timestamps."""
    
    def __init__(self, max_commands: int = 5):
        """Initialize command history with maximum size."""
        self.commands = []
        self.max_commands = max_commands
        self.lock = Lock()
    
    def add_command(self, command_info: Dict[str, Any]) -> None:
        """Add a new command to the history."""
        with self.lock:
            # Create a simpler version for display
            display_command = {
                "speaker": command_info["speaker"],
                "text": command_info["transcript"],
                "time": time.strftime("%H:%M:%S", time.localtime(command_info["timestamp"])),
                "confidence": command_info["confidence"]
            }
            
            # Add to history
            self.commands.append(display_command)
            
            # Trim if needed
            if len(self.commands) > self.max_commands:
                self.commands = self.commands[-self.max_commands:]
    
    def get_commands(self) -> List[Dict[str, Any]]:
        """Get the current command history."""
        with self.lock:
            return list(self.commands)


# Main application class
class SpeakerVerificationApp:
    """
    Demonstration application for continuous speaker verification.
    
    Features:
    - Real-time speaker verification display
    - Command history visualization
    - Audio level monitoring
    """
    
    def __init__(self, enrollment_dir: str = "enrollments"):
        """
        Initialize the application.
        
        Args:
            enrollment_dir: Directory containing speaker enrollments
        """
        # Create components
        self.command_history = CommandHistory()
        self.monitor = ContinuousSpeakerMonitor()
        
        # Load existing enrollments
        self.voice_recognizer = self.monitor.voice_recognizer
        self.load_enrollments(enrollment_dir)
        
        # UI state
        self.current_speaker = None
        self.confidence = 0.0
        self.audio_levels = []
        self.max_audio_points = 100
        self.speech_active = False
        
        # Configure monitor callbacks
        self.monitor.add_speaker_change_callback(self.on_speaker_change)
        self.monitor.add_speech_command_callback(self.on_command)
    
    def load_enrollments(self, enrollment_dir: str) -> None:
        """
        Load speaker enrollments from the specified directory.
        
        Args:
            enrollment_dir: Directory containing enrollment files
        """
        if os.path.exists(enrollment_dir):
            # Load enrollments with the voice recognizer
            num_loaded = self.voice_recognizer.load_speaker_embeddings(enrollment_dir)
            print(f"Loaded {num_loaded} speaker enrollments")
            
            if num_loaded == 0:
                print(f"No enrollments found in {enrollment_dir}")
        else:
            print(f"Enrollment directory {enrollment_dir} not found")
            
            # Create directory
            try:
                os.makedirs(enrollment_dir)
                print(f"Created enrollment directory: {enrollment_dir}")
            except Exception as e:
                print(f"Error creating enrollment directory: {e}")
    
    def on_speaker_change(self, change_info: Dict[str, Any]) -> None:
        """
        Handle speaker change events.
        
        Args:
            change_info: Information about the speaker change
        """
        self.current_speaker = change_info.get("current_speaker")
        self.confidence = change_info.get("confidence", 0.0)
        
        # Print change info
        prev = change_info.get("previous_speaker", "None")
        curr = self.current_speaker or "None"
        print(f"Speaker change: {prev} -> {curr} (confidence: {self.confidence:.2f})")
    
    def on_command(self, command_info: Dict[str, Any]) -> None:
        """
        Handle speech command events.
        
        Args:
            command_info: Information about the speech command
        """
        # Add to history
        self.command_history.add_command(command_info)
        
        # Print command
        speaker = command_info.get("speaker", "Unknown")
        transcript = command_info.get("transcript", "")
        print(f"Command from {speaker}: {transcript}")
    
    def on_audio_chunk(self, audio_data: np.ndarray) -> None:
        """
        Handle audio chunk events for visualization.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Calculate audio level
        level = np.abs(audio_data).mean()
        self.audio_levels.append(level)
        
        # Trim if needed
        if len(self.audio_levels) > self.max_audio_points:
            self.audio_levels = self.audio_levels[-self.max_audio_points:]
    
    def update_ui(self, frame: int, 
                  speaker_text, command_list, 
                  audio_line, speech_indicator) -> List:
        """
        Update the UI elements.
        
        Args:
            frame: Animation frame
            speaker_text: Text element for speaker info
            command_list: Text element for command list
            audio_line: Line for audio levels
            speech_indicator: Indicator for speech activity
        
        Returns:
            List of updated artists
        """
        # Update speaker info
        if self.current_speaker:
            speaker_text.set_text(f"Speaker: {self.current_speaker}\nConfidence: {self.confidence:.2f}")
            speaker_text.set_color('green')
        else:
            speaker_text.set_text("No speaker verified")
            speaker_text.set_color('red')
        
        # Update command list
        commands = self.command_history.get_commands()
        if commands:
            cmd_text = "\n".join([
                f"[{cmd['time']}] {cmd['speaker']}: {cmd['text']}"
                for cmd in reversed(commands)
            ])
            command_list.set_text(cmd_text)
        else:
            command_list.set_text("No commands yet")
        
        # Update audio levels
        if self.audio_levels:
            audio_line.set_ydata(self.audio_levels)
        
        # Update speech indicator
        speech_indicator.set_color('green' if self.speech_active else 'red')
        speech_indicator.set_text("SPEECH" if self.speech_active else "SILENCE")
        
        return [speaker_text, command_list, audio_line, speech_indicator]
    
    def run(self) -> None:
        """Run the application with visualization."""
        # Set up the figure
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.manager.set_window_title("Continuous Speaker Verification Demo")
        
        # Speaker verification area
        speaker_ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
        speaker_ax.axis('off')
        speaker_text = speaker_ax.text(0.5, 0.5, "No speaker verified", 
                                    ha='center', va='center', fontsize=24,
                                    transform=speaker_ax.transAxes)
        
        # Speech activity indicator
        speech_ax = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
        speech_ax.axis('off')
        speech_indicator = speech_ax.text(0.5, 0.5, "SILENCE", 
                                       ha='center', va='center', fontsize=24,
                                       transform=speech_ax.transAxes)
        speech_indicator.set_color('red')
        
        # Audio level visualization
        audio_ax = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=1)
        audio_ax.set_title("Audio Level")
        audio_ax.set_ylim(0, 0.5)
        audio_ax.set_xlim(0, self.max_audio_points)
        audio_line, = audio_ax.plot(np.zeros(self.max_audio_points))
        
        # Command history area
        cmd_ax = plt.subplot2grid((3, 3), (2, 0), colspan=3, rowspan=1)
        cmd_ax.axis('off')
        cmd_ax.set_title("Command History")
        command_list = cmd_ax.text(0.05, 0.95, "No commands yet",
                                 va='top', transform=cmd_ax.transAxes,
                                 fontfamily='monospace')
        
        # Register audio callback for visualization
        def audio_viz_callback(audio_data):
            self.on_audio_chunk(audio_data)
        
        self.monitor.audio_stream.register_callback(audio_viz_callback)
        
        # Register VAD state callback
        def vad_state_callback(is_speech, prob):
            self.speech_active = is_speech
        
        self.monitor.vad.register_state_callback(vad_state_callback)
        
        # Start monitoring
        self.monitor.start()
        
        # Create animation for UI updates
        ani = FuncAnimation(fig, self.update_ui, fargs=(
            speaker_text, command_list, audio_line, speech_indicator
        ), interval=100, blit=True)
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        # Clean up
        self.monitor.stop()


# Enrollment class for adding new speakers
class EnrollmentApp:
    """Application for enrolling new speakers."""
    
    def __init__(self, enrollment_dir: str = "enrollments"):
        """
        Initialize the enrollment application.
        
        Args:
            enrollment_dir: Directory to store enrollments
        """
        self.enrollment_dir = enrollment_dir
        self.voice_recognizer = VoiceRecognizer()
        
        # Make sure directory exists
        if not os.path.exists(enrollment_dir):
            os.makedirs(enrollment_dir)
    
    def enroll_speaker(self, speaker_name: str) -> bool:
        """
        Enroll a new speaker with recording.
        
        Args:
            speaker_name: Name of the speaker to enroll
            
        Returns:
            bool: True if enrollment was successful
        """
        print(f"Enrolling new speaker: {speaker_name}")
        print("Please speak for 5 seconds when prompted...")
        time.sleep(1)
        
        print("Recording will start in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("Recording... Please speak naturally.")
        
        try:
            # Perform enrollment
            result = self.voice_recognizer.enroll_speaker(speaker_name)
            
            if result:
                print(f"Successfully enrolled {speaker_name}!")
                
                # Save embeddings
                self.voice_recognizer.save_speaker_embeddings(self.enrollment_dir)
                return True
            else:
                print(f"Failed to enroll {speaker_name}")
                return False
                
        except Exception as e:
            print(f"Error during enrollment: {e}")
            return False


# Main entry point
def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Speaker Verification Demo")
    parser.add_argument('--enroll', action='store_true', help='Enroll a new speaker')
    parser.add_argument('--name', type=str, help='Name of the speaker to enroll')
    parser.add_argument('--dir', type=str, default='enrollments',
                       help='Directory for storing enrollments')
    
    args = parser.parse_args()
    
    if args.enroll:
        if not args.name:
            print("Error: Speaker name required for enrollment")
            parser.print_help()
            return
        
        # Run enrollment
        enrollment_app = EnrollmentApp(args.dir)
        enrollment_app.enroll_speaker(args.name)
    else:
        # Run demo application
        app = SpeakerVerificationApp(args.dir)
        try:
            app.run()
        except KeyboardInterrupt:
            print("\nExiting...")


# Run the application when executed directly
if __name__ == "__main__":
    main() 