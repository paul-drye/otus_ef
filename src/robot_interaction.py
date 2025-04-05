import yaml
import pyttsx3
import time


class RobotInteraction:
    def __init__(self, config_path='config/settings.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.tts_config = self.config['robot_interaction']
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        # Configure voice properties
        self.tts_engine.setProperty('rate', self.tts_config['voice_rate'])
        self.tts_engine.setProperty('volume', self.tts_config['voice_volume'])
        
        # Try to set a voice
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to find a female voice (usually index 1)
            voice_idx = 1 if len(voices) > 1 else 0
            self.tts_engine.setProperty('voice', voices[voice_idx].id)
    
    def speak(self, text):
        """
        Convert text to speech.
        
        Args:
            text (str): Text to be spoken
        """
        print(f"Robot: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def greet_user(self, user_name=None):
        """
        Greet the user.
        
        Args:
            user_name (str, optional): Name of the user to greet
        """
        if user_name:
            self.speak(f"Hello, {user_name}! How can I help you today?")
        else:
            self.speak("Hello! I don't recognize you. Please identify yourself.")
    
    def deny_access(self):
        """
        Notify that access is denied.
        """
        self.speak("Access denied. I could not verify your identity.")
    
    def authentication_success(self, user_name):
        """
        Notify that authentication was successful.
        
        Args:
            user_name (str): Name of the authenticated user
        """
        self.speak(f"Authentication successful. Welcome, {user_name}!")
    
    def authentication_failed(self, reason=None):
        """
        Notify that authentication failed.
        
        Args:
            reason (str, optional): Reason for failure
        """
        if reason:
            self.speak(f"Authentication failed. {reason}")
        else:
            self.speak("Authentication failed. Please try again.")
    
    def prompt_face(self):
        """
        Prompt user to show their face.
        """
        self.speak("Please look at the camera so I can see your face.")
    
    def prompt_voice(self):
        """
        Prompt user to speak.
        """
        self.speak("Please say something so I can verify your voice.")
    
    def enrollment_instruction(self, type_str):
        """
        Give instructions for enrollment.
        
        Args:
            type_str (str): Type of enrollment (face/voice)
        """
        if type_str.lower() == 'face':
            self.speak("I will now capture your face. Please look directly at the camera.")
        elif type_str.lower() == 'voice':
            self.speak("I will now record your voice. Please speak clearly.")
        else:
            self.speak("I need to register your biometric data. Please follow the instructions.")
    
    def enrollment_success(self):
        """
        Notify that enrollment was successful.
        """
        self.speak("Enrollment successful. I can now recognize you.")
    
    def enrollment_failed(self):
        """
        Notify that enrollment failed.
        """
        self.speak("Enrollment failed. Please try again.")
    
    def respond_to_command(self, command):
        """
        Respond to a user command.
        
        Args:
            command (str): Command from the user
        
        Returns:
            bool: True if valid command was recognized, False otherwise
        """
        # Simple command handling
        command = command.lower()
        
        if 'hello' in command or 'hi' in command:
            self.speak("Hello there!")
            return True
        elif 'how are you' in command:
            self.speak("I'm functioning normally, thank you for asking.")
            return True
        elif 'name' in command:
            self.speak("I'm your authentication robot assistant.")
            return True
        elif 'thank' in command:
            self.speak("You're welcome!")
            return True
        elif 'bye' in command or 'goodbye' in command:
            self.speak("Goodbye! Have a great day.")
            return True
        else:
            return False


if __name__ == "__main__":
    # Simple test
    robot = RobotInteraction()
    
    # Test different utterances
    robot.greet_user()
    time.sleep(1)
    
    robot.greet_user("John")
    time.sleep(1)
    
    robot.prompt_face()
    time.sleep(1)
    
    robot.prompt_voice()
    time.sleep(1)
    
    robot.authentication_success("John")
    time.sleep(1)
    
    robot.authentication_failed("Face not recognized")
    time.sleep(1)
    
    robot.deny_access()
    time.sleep(1)
    
    # Test command responses
    robot.respond_to_command("Hello there!")
    time.sleep(1)
    
    robot.respond_to_command("What's your name?")
    time.sleep(1)
    
    robot.respond_to_command("How are you doing?")
    time.sleep(1)
    
    robot.respond_to_command("Thank you for your help")
    time.sleep(1)
    
    robot.respond_to_command("Goodbye")
