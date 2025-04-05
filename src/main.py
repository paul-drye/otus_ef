#!/usr/bin/env python3
import os
import sys
import argparse
import time

from decision_engine import DecisionEngine
from robot_interaction import RobotInteraction


def create_user(decision_engine, robot, args):
    """Enroll a new user with face and voice."""
    robot.speak("Starting user enrollment process.")
    robot.enrollment_instruction("face")
    
    time.sleep(1)
    success = decision_engine.enroll_user(
        args.name, 
        face_samples=args.face_samples, 
        voice_samples=args.voice_samples,
        confirm=not args.no_confirm
    )
    
    if success:
        robot.enrollment_success()
    else:
        robot.enrollment_failed()
    
    return success


def authenticate(decision_engine, robot, args):
    """Authenticate a user."""
    robot.speak("Starting authentication process.")
    
    # Prompt for face
    robot.prompt_face()
    time.sleep(1)
    
    # Authenticate
    authenticated, user_name = decision_engine.authenticate_user(
        face_time=args.face_time, display_video=args.display_video
    )
    
    if authenticated:
        robot.authentication_success(user_name)
        return True, user_name
    else:
        robot.authentication_failed()
        return False, None


def interactive_mode(decision_engine, robot):
    """Run in interactive mode, allowing command input after authentication."""
    authenticated, user_name = authenticate(decision_engine, robot, parse_args(['auth']))
    
    if not authenticated:
        return
    
    robot.speak("You are now in interactive mode. Speak commands or say 'exit' to quit.")
    
    while True:
        # Get speech input
        print("\nListening for command...")
        command = decision_engine.get_user_speech_input()
        
        if not command:
            robot.speak("I didn't catch that. Could you please repeat?")
            continue
        
        print(f"Command: {command}")
        
        # Check for exit command
        if 'exit' in command.lower() or 'quit' in command.lower():
            robot.speak("Exiting interactive mode. Goodbye!")
            break
        
        # Try to respond to the command
        if not robot.respond_to_command(command):
            robot.speak("I'm not sure how to respond to that command.")


def list_users(decision_engine, robot):
    """List all enrolled users."""
    face_users = decision_engine.face_recognizer.known_face_names
    voice_users = list(decision_engine.voice_recognizer.speaker_embeddings.keys())
    
    print("\n=== Enrolled Users ===")
    print("Face recognition users:")
    for user in face_users:
        print(f"- {user}")
    
    print("\nVoice recognition users:")
    for user in voice_users:
        print(f"- {user}")
    
    # Find users enrolled with both face and voice
    common_users = set(face_users).intersection(set(voice_users))
    print("\nFully enrolled users (face + voice):")
    for user in common_users:
        print(f"- {user}")
    
    if common_users:
        robot.speak(f"There are {len(common_users)} fully enrolled users in the system.")
    else:
        robot.speak("There are no fully enrolled users in the system.")


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face and Voice Authentication System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', help='Enroll a new user')
    enroll_parser.add_argument('--name', '-n', required=True, help='Name of the user to enroll')
    enroll_parser.add_argument('--face-samples', type=int, default=5, 
                               help='Number of face samples to capture')
    enroll_parser.add_argument('--voice-samples', type=int, default=1, 
                               help='Number of voice samples to record (always 1, kept for backward compatibility)')
    enroll_parser.add_argument('--no-confirm', action='store_true',
                               help='Skip confirmation steps during enrollment')
    
    # Authenticate command
    auth_parser = subparsers.add_parser('auth', help='Authenticate a user')
    auth_parser.add_argument('--face-time', type=int, default=5, 
                            help='Time in seconds to attempt face recognition')
    auth_parser.add_argument('--display-video', action='store_true', 
                            help='Display video feed during face recognition')
    
    # Interactive mode command
    subparsers.add_parser('interactive', help='Run in interactive mode after authentication')
    
    # List users command
    subparsers.add_parser('list-users', help='List all enrolled users')
    
    return parser.parse_args(args)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create decision engine and robot interaction
    decision_engine = DecisionEngine()
    robot = RobotInteraction()
    
    if args.command == 'enroll':
        create_user(decision_engine, robot, args)
    elif args.command == 'auth':
        authenticate(decision_engine, robot, args)
    elif args.command == 'interactive':
        interactive_mode(decision_engine, robot)
    elif args.command == 'list-users':
        list_users(decision_engine, robot)
    else:
        # Display help if no command is provided
        print("Please specify a command.")
        robot.speak("Please specify a command such as enroll, auth, or interactive.")


if __name__ == "__main__":
    main()
