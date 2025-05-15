#!/usr/bin/env python
"""
Test script for validating the updated ElevenLabs transcription pipeline with real API responses.
This script takes two audio files (one for customer, one for AI agent) and tests the transcription
and overlap detection functionality using the real ElevenLabs API.
"""

import os
import tempfile
import argparse
from dotenv import load_dotenv
from audio_metrics import AudioMetricsCalculator


def test_real_transcription(customer_file, agent_file):
    """Test the transcription pipeline with real audio files."""
    print(f"\n=== Testing Transcription with Real Files ===")
    print(f"Customer audio: {customer_file}")
    print(f"Agent audio: {agent_file}")
    
    # Load environment variables to get ElevenLabs API key
    load_dotenv()
    
    # Create a calculator with the real ElevenLabs client
    calculator = AudioMetricsCalculator()
    
    if not calculator.elevenlabs_client:
        print("ERROR: ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in your .env file.")
        return
    
    # Create a temporary directory to store the processed files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Get absolute paths for the audio files
        customer_path = os.path.abspath(customer_file) if os.path.exists(customer_file) else None
        agent_path = os.path.abspath(agent_file) if os.path.exists(agent_file) else None
        
        if not customer_path or not agent_path:
            print("ERROR: Could not find one or both of the audio files.")
            return
        
        # Split and transcribe
        print("\nTranscribing customer audio...")
        customer_words = calculator._transcribe_channel(customer_path, "customer")
        print(f"Found {len(customer_words)} words in customer audio")
        
        print("\nTranscribing agent audio...")
        agent_words = calculator._transcribe_channel(agent_path, "ai_agent")
        print(f"Found {len(agent_words)} words in agent audio")
        
        if not customer_words and not agent_words:
            print("ERROR: Transcription failed for both audio files.")
            return
        
        print("\nMerging transcripts and detecting overlaps...")
        transcript = calculator._merge_and_format_transcript(customer_words, agent_words)
        
        # Print overlap information
        print(f"\n=== Overlap Analysis Results ===")
        print(f"Has overlaps: {transcript['has_overlaps']}")
        print(f"Overlap count: {transcript['overlap_count']}")
        
        # Print some overlapping words
        overlapping_words = [w for w in transcript['words'] if w.get('is_overlap', False)]
        if overlapping_words:
            print("\nSample overlapping words:")
            for i, word in enumerate(overlapping_words[:5]):  # Show up to 5 examples
                print(f"  - '{word['text']}' ({word['speaker']}): {word['start']:.2f}s to {word['end']:.2f}s")
        
        # Print formatted transcript
        print("\n=== Formatted Transcript ===")
        print(transcript['dialog'])
        
        print("\n=== Test Completed Successfully ===")


def main():
    parser = argparse.ArgumentParser(description='Test ElevenLabs transcription with real audio files.')
    parser.add_argument('--customer', '-c', required=True, help='Path to customer audio file (mono)')
    parser.add_argument('--agent', '-a', required=True, help='Path to agent audio file (mono)')
    
    args = parser.parse_args()
    test_real_transcription(args.customer, args.agent)


if __name__ == "__main__":
    main()
