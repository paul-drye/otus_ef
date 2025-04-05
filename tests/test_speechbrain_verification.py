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
    if audio_file is None or not os.path.exists(audio_file):
        print("No audio file provided. Recording a sample...")
        audio_file = recognizer.record_audio(duration=5)
    
    # Extract embedding
    embedding = recognizer.extract_speaker_embedding(audio_file)
    
    if embedding is not None:
        print(f"Successfully extracted embedding with shape: {embedding.shape}")
        
        # Clean up temporary audio file if we created it
        if audio_file is None and os.path.exists(audio_file):
            os.remove(audio_file)
            
        return True
    else:
        print("Failed to extract embedding.")
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


def test_similarity_calculation():
    """Test the similarity calculation between two embeddings."""
    print("Testing similarity calculation...")
    
    recognizer = VoiceRecognizer()
    
    # Create two random embeddings
    emb1 = torch.rand(192)  # ECAPA-TDNN produces 192-dimensional embeddings
    emb2 = torch.rand(192)
    
    # Calculate similarity
    similarity = recognizer._compute_similarity(emb1, emb2)
    
    print(f"Similarity between random embeddings: {similarity:.4f}")
    print("This should be close to 0 for random vectors")
    
    # Test with same embedding
    similarity = recognizer._compute_similarity(emb1, emb1)
    
    print(f"Similarity with self: {similarity:.4f}")
    print("This should be close to 1.0")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test SpeechBrain Speaker Verification')
    
    subparsers = parser.add_subparsers(dest='command', help='Test command')
    
    # Model loading test
    subparsers.add_parser('model', help='Test model loading')
    
    # Embedding test
    embedding_parser = subparsers.add_parser('embedding', help='Test embedding extraction')
    embedding_parser.add_argument('--audio', type=str, help='Path to audio file')
    
    # Enrollment test
    enrollment_parser = subparsers.add_parser('enroll', help='Test user enrollment')
    enrollment_parser.add_argument('--name', type=str, default="Test User", help='Name for the test user')
    
    # Verification test
    verification_parser = subparsers.add_parser('verify', help='Test speaker verification')
    verification_parser.add_argument('--enroll-first', action='store_true', help='Enroll a test user first')
    
    # Similarity test
    subparsers.add_parser('similarity', help='Test similarity calculation')
    
    # All tests
    subparsers.add_parser('all', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.command == 'model':
        test_speechbrain_model_loading()
    
    elif args.command == 'embedding':
        test_embedding_extraction(args.audio)
    
    elif args.command == 'enroll':
        test_user_enrollment(args.name)
    
    elif args.command == 'verify':
        test_speaker_verification(args.enroll_first)
    
    elif args.command == 'similarity':
        test_similarity_calculation()
    
    elif args.command == 'all':
        print("Running all tests...")
        test_speechbrain_model_loading()
        test_embedding_extraction()
        test_similarity_calculation()
        test_user_enrollment()
        test_speaker_verification()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 