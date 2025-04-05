#!/usr/bin/env python3
import os
import sys
import argparse
import time
import torch

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.voice_recognition_module import VoiceRecognizer


def test_speechbrain_model_loading():
    """Test if the SpeechBrain model loads successfully."""
    print("Testing SpeechBrain model loading...")
    
    recognizer = VoiceRecognizer()
    
    if recognizer.model_loaded:
        print("SpeechBrain model loaded successfully!")
        return True
    else:
        print("Failed to load SpeechBrain model.")
        return False


def test_embedding_extraction(audio_file=None):
    """
    Test embedding extraction using SpeechBrain.
    
    Args:
        audio_file (str, optional): Path to an audio file to use for testing
    """
    print("Testing embedding extraction...")
    
    recognizer = VoiceRecognizer()
    
    if not recognizer.model_loaded:
        print("SpeechBrain model not loaded. Cannot test embedding extraction.")
        return False
    
    # If no audio file provided, record one
    temp_file_created = False
    if audio_file is None or not os.path.exists(audio_file):
        print("No audio file provided. Recording a sample...")
        audio_file = recognizer.record_audio(duration=5)
        temp_file_created = True
    
    # Extract embedding
    embedding = recognizer.extract_speaker_embedding(audio_file)
    
    # Clean up temporary audio file if we created it
    if temp_file_created and os.path.exists(audio_file):
        os.remove(audio_file)
    
    if embedding is not None:
        print(f"Successfully extracted embedding with shape: {embedding.shape}")
        return True
    else:
        print("Failed to extract embedding.")
        return False


def test_embedding_comparison(audio_file1=None, audio_file2=None):
    """
    Test direct comparison between two speech embeddings.
    
    Args:
        audio_file1 (str, optional): Path to the first audio file
        audio_file2 (str, optional): Path to the second audio file
    """
    print("Testing direct comparison between two speech embeddings...")
    
    recognizer = VoiceRecognizer()
    
    if not recognizer.model_loaded:
        print("SpeechBrain model not loaded. Cannot test embedding comparison.")
        return False
    
    # If no audio files provided, record them
    temp_files = []
    
    if audio_file1 is None or not os.path.exists(audio_file1):
        print("Recording first audio sample...")
        audio_file1 = recognizer.record_audio(duration=5)
        temp_files.append(audio_file1)
    
    if audio_file2 is None or not os.path.exists(audio_file2):
        print("Recording second audio sample (may be same or different speaker)...")
        audio_file2 = recognizer.record_audio(duration=5)
        temp_files.append(audio_file2)
    
    # Extract embeddings
    embedding1 = recognizer.extract_speaker_embedding(audio_file1)
    embedding2 = recognizer.extract_speaker_embedding(audio_file2)
    
    # Clean up temporary audio files
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
    
    if embedding1 is None or embedding2 is None:
        print("Failed to extract one or both embeddings.")
        return False
    
    # Compare embeddings using the verification model's similarity function
    try:
        # Make sure embeddings are tensors with the right shape
        if isinstance(embedding1, torch.Tensor):
            emb1 = embedding1.unsqueeze(0)
        else:
            emb1 = torch.tensor(embedding1).unsqueeze(0)
            
        if isinstance(embedding2, torch.Tensor):
            emb2 = embedding2.unsqueeze(0)
        else:
            emb2 = torch.tensor(embedding2).unsqueeze(0)
        
        score = recognizer.verification_model.similarity(emb1, emb2)
        score = score.item()
        
        prediction = score > recognizer.verification_threshold
        
        print(f"Similarity score: {score:.4f} (Threshold: {recognizer.verification_threshold})")
        print(f"Same speaker prediction: {prediction}")
        
        return True
    except Exception as e:
        print(f"Error comparing embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_user_enrollment(user_name="Test User"):
    """
    Test user enrollment with SpeechBrain.
    
    Args:
        user_name (str): Name for the test user
    """
    print(f"Testing user enrollment for '{user_name}'...")
    
    recognizer = VoiceRecognizer()
    
    if not recognizer.model_loaded:
        print("SpeechBrain model not loaded. Cannot test enrollment.")
        return False
    
    # Enroll user
    success = recognizer.enroll_user(user_name)
    
    if success:
        print(f"Successfully enrolled user '{user_name}'")
        
        # Verify the user's embedding was saved
        if user_name in recognizer.speaker_embeddings:
            embedding = recognizer.speaker_embeddings[user_name]
            print(f"User embedding saved, shape: {embedding.shape}")
        else:
            print("Warning: User embedding not found")
        
        return True
    else:
        print(f"Failed to enroll user '{user_name}'")
        return False


def test_speaker_verification(enroll_first=False):
    """
    Test speaker verification with SpeechBrain.
    
    Args:
        enroll_first (bool): Whether to enroll a test user first
    """
    print("Testing speaker verification...")
    
    recognizer = VoiceRecognizer()
    
    if not recognizer.model_loaded:
        print("SpeechBrain model not loaded. Cannot test verification.")
        return False
    
    # Check if there are any enrolled users
    if not recognizer.speaker_embeddings and not enroll_first:
        print("No enrolled users found. Please enroll a user first or set enroll_first=True.")
        return False
    
    # Enroll a test user if requested
    if enroll_first:
        test_user_enrollment("Test User")
    
    # Verify speaker
    print("Please speak for verification...")
    verified, name = recognizer.verify_speaker()
    
    if verified:
        print(f"Speaker verified as '{name}' using SpeechBrain!")
        return True
    else:
        print("Speaker verification failed.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test SpeechBrain Speaker Verification')
    
    subparsers = parser.add_subparsers(dest='command', help='Test command')
    
    # Model loading test
    subparsers.add_parser('model', help='Test model loading')
    
    # Embedding extraction test
    embedding_parser = subparsers.add_parser('embedding', help='Test embedding extraction')
    embedding_parser.add_argument('--audio', type=str, help='Path to audio file')
    
    # Embedding comparison test
    comparison_parser = subparsers.add_parser('compare', help='Test embedding comparison')
    comparison_parser.add_argument('--file1', type=str, help='Path to first audio file')
    comparison_parser.add_argument('--file2', type=str, help='Path to second audio file')
    
    # Enrollment test
    enrollment_parser = subparsers.add_parser('enroll', help='Test user enrollment')
    enrollment_parser.add_argument('--name', type=str, default="Test User", help='Name for the test user')
    
    # Verification test
    verification_parser = subparsers.add_parser('verify', help='Test speaker verification')
    verification_parser.add_argument('--enroll-first', action='store_true', help='Enroll a test user first')
    
    # All tests
    subparsers.add_parser('all', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.command == 'model':
        test_speechbrain_model_loading()
    
    elif args.command == 'embedding':
        test_embedding_extraction(args.audio)
    
    elif args.command == 'compare':
        test_embedding_comparison(args.file1, args.file2)
    
    elif args.command == 'enroll':
        test_user_enrollment(args.name)
    
    elif args.command == 'verify':
        test_speaker_verification(args.enroll_first)
    
    elif args.command == 'all':
        print("Running all tests...")
        test_speechbrain_model_loading()
        test_embedding_extraction()
        test_embedding_comparison()
        test_user_enrollment()
        test_speaker_verification()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 