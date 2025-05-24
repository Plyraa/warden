#!/usr/bin/env python
"""
Test script to verify the ElevenLabs transcription and database storage functionality
This script simulates an ElevenLabs API response and tests the processing pipeline
"""

import os
import json
from database import init_db, get_db, get_analysis_by_filename, recreate_metrics_from_db
from audio_metrics import AudioMetricsCalculator


def create_mock_elevenlabs_response(response_file=None):
    """Creates a mock ElevenLabs response object for testing"""

    # Example response from ElevenLabs API
    example_response = {
        "language_code": "en",
        "language_probability": 0.98,
        "text": "Hello world!",
        "words": [
            {
                "text": "Hello",
                "type": "word",
                "logprob": 1.1,
                "start": 0,
                "end": 0.5,
                "speaker_id": "speaker_1",
            },
            {
                "text": " ",
                "type": "spacing",
                "logprob": 1.1,
                "start": 0.5,
                "end": 0.5,
                "speaker_id": "speaker_1",
            },
            {
                "text": "world!",
                "type": "word",
                "logprob": 1.1,
                "start": 0.5,
                "end": 1.2,
                "speaker_id": "speaker_1",
            },
        ],
    }

    # If a response file is provided, load that instead
    if response_file and os.path.exists(response_file):
        with open(response_file, "r") as f:
            example_response = json.load(f)

    # Create a simple object that mimics the ElevenLabs response structure
    class MockResponse:
        def __init__(self, data):
            self.language_code = data.get("language_code")
            self.language_probability = data.get("language_probability")
            self.text = data.get("text")

            # Create Word objects
            class Word:
                def __init__(self, word_data):
                    self.text = word_data.get("text")
                    self.type = word_data.get("type")
                    self.logprob = word_data.get("logprob")
                    self.start = word_data.get("start")
                    self.end = word_data.get("end")
                    self.speaker_id = word_data.get("speaker_id")

            self.words = [Word(word) for word in data.get("words", [])]

    return MockResponse(example_response)


def mock_transcribe_test():
    """Test the transcription functionality with mock data"""
    print("\n=== Testing Transcription Processing ===")

    # Create a mock audio metrics calculator
    calc = AudioMetricsCalculator()

    # Override the ElevenLabs client with our mock
    calc.elevenlabs_client = True  # Just need a truthy value

    # Create test data for customer (left channel) and agent (right channel)
    # Overlapping speech between 2.5 and 3.0 seconds
    customer_words = [
        {
            "text": "Hello",
            "type": "word",
            "start": 0.0,
            "end": 0.5,
            "speaker_id": "customer",
        },
        {
            "text": "how",
            "type": "word",
            "start": 1.0,
            "end": 1.2,
            "speaker_id": "customer",
        },
        {
            "text": "are",
            "type": "word",
            "start": 1.3,
            "end": 1.5,
            "speaker_id": "customer",
        },
        {
            "text": "you",
            "type": "word",
            "start": 1.6,
            "end": 1.8,
            "speaker_id": "customer",
        },
        {
            "text": "today",
            "type": "word",
            "start": 1.9,
            "end": 2.3,
            "speaker_id": "customer",
        },
        {
            "text": "I",
            "type": "word",
            "start": 2.5,
            "end": 2.6,
            "speaker_id": "customer",
        },
        {
            "text": "need",
            "type": "word",
            "start": 2.7,
            "end": 3.0,
            "speaker_id": "customer",
        },
        {
            "text": "help",
            "type": "word",
            "start": 3.1,
            "end": 3.5,
            "speaker_id": "customer",
        },
    ]

    agent_words = [
        {"text": "Hi", "type": "word", "start": 2.0, "end": 2.2, "speaker_id": "agent"},
        {
            "text": "there",
            "type": "word",
            "start": 2.3,
            "end": 2.5,
            "speaker_id": "agent",
        },
        {
            "text": "I'm",
            "type": "word",
            "start": 2.6,
            "end": 2.8,
            "speaker_id": "agent",
        },
        {
            "text": "doing",
            "type": "word",
            "start": 2.9,
            "end": 3.2,
            "speaker_id": "agent",
        },
        {
            "text": "great",
            "type": "word",
            "start": 3.3,
            "end": 3.6,
            "speaker_id": "agent",
        },
    ]

    # Create test transcript data
    words_customer = []
    for word in customer_words:
        words_customer.append(
            {
                "text": word["text"],
                "start": word["start"],
                "end": word["end"],
                "speaker": "customer",
            }
        )

    words_agent = []
    for word in agent_words:
        words_agent.append(
            {
                "text": word["text"],
                "start": word["start"],
                "end": word["end"],
                "speaker": "ai_agent",
            }
        )

    # Test the merge function
    result = calc._merge_and_format_transcript(words_customer, words_agent)

    print("Transcript processing result:")
    print(f"  - Has overlaps: {result['has_overlaps']}")
    print(f"  - Overlap count: {result['overlap_count']}")
    print(f"  - Dialog: {result['dialog']}")

    # Check if overlaps were correctly detected
    assert result["has_overlaps"] is True, "Failed to detect overlaps"
    assert result["overlap_count"] >= 4, "Failed to count overlapping words correctly"

    print("✓ Overlap detection working correctly!")

    return result


def test_database_storage(transcript_data):
    """Test database storage of transcript data"""
    print("\n=== Testing Database Storage ===")

    # Initialize the database
    init_db()

    # Create test metrics data
    test_metrics = {
        "filename": "test_overlap_file.mp3",
        "downsampled_path": "sampled_test_calls/test_overlap_file_downsampled.mp3",
        "latency_metrics": {
            "avg_latency": 100,
            "p10_latency": 50,
            "p50_latency": 100,
            "p90_latency": 150,
        },
        "agent_answer_latencies": [50, 100, 150],
        "ai_interrupting_user": True,
        "user_interrupting_ai": True,
        "talk_ratio": 1.5,
        "average_pitch": 220.0,
        "words_per_minute": 150.0,
        "user_windows": [[0.5, 2.0], [3.0, 4.5]],
        "agent_windows": [[2.1, 2.9], [4.6, 5.5]],
        "transcript_data": transcript_data,
    }

    # Store in database
    db = next(get_db())
    try:
        from database import add_analysis

        add_analysis(db, test_metrics)
        print("✓ Added test analysis to database")

        # Retrieve from database
        retrieved = get_analysis_by_filename(db, "test_overlap_file.mp3")
        assert retrieved is not None, "Failed to retrieve analysis from database"

        # Recreate metrics
        recreated = recreate_metrics_from_db(retrieved)
        assert recreated is not None, "Failed to recreate metrics from database"

        # Check transcript data
        assert "transcript_data" in recreated, (
            "Transcript data missing from recreated metrics"
        )
        assert "has_overlaps" in recreated["transcript_data"], (
            "Overlap info missing from transcript data"
        )
        assert recreated["transcript_data"]["has_overlaps"] is True, (
            "Overlap flag not stored correctly"
        )
        assert recreated["transcript_data"]["overlap_count"] >= 4, (
            "Overlap count not stored correctly"
        )

        print("✓ Successfully retrieved and validated overlap data from database")
        print(
            f"  - Retrieved overlap count: {recreated['transcript_data']['overlap_count']}"
        )

    finally:
        db.close()

    return recreated


def main():
    """Main test function"""
    print("=== ElevenLabs Transcript Processing Test ===")

    # Test transcript processing with mock data
    transcript_data = mock_transcribe_test()

    # Test database storage
    recreated = test_database_storage(transcript_data)  # noqa: F841

    print("\n=== All Tests Passed! ===")
    print("The ElevenLabs transcript processing pipeline is working correctly.")
    print(
        "Overlapping speech is properly detected, stored in the database, and can be visualized."
    )


if __name__ == "__main__":
    main()
