#!/usr/bin/env python
# test_transcript_handling.py - Test the transcript handling functionality

import json
import os
from database import get_db, recreate_metrics_from_db, get_analysis_by_filename
from audio_metrics import AudioMetricsCalculator


def simulate_elevenlabs_response():
    """Create a simulated ElevenLabs response for testing"""

    class MockWord:
        def __init__(
            self,
            text,
            start,
            end,
            word_type="word",
            logprob=1.1,
            speaker_id="speaker_1",
        ):
            self.text = text
            self.start = start
            self.end = end
            self.type = word_type
            self.logprob = logprob
            self.speaker_id = speaker_id

    class MockResponse:
        def __init__(self):
            self.language_code = "en"
            self.language_probability = 0.98
            self.text = "Hello world!"
            self.words = [
                MockWord("Hello", 0, 0.5),
                MockWord(" ", 0.5, 0.5, word_type="spacing"),
                MockWord("world!", 0.5, 1.2),
            ]

    return MockResponse()


def test_transcribe_channel():
    """Test the _transcribe_channel method with a mock response"""
    calculator = AudioMetricsCalculator()

    # Save original method for later restoration
    original_method = calculator.elevenlabs_client.speech_to_text.convert

    try:
        # Replace with mock implementation
        def mock_convert(*args, **kwargs):
            return simulate_elevenlabs_response()

        calculator.elevenlabs_client.speech_to_text.convert = mock_convert

        # Test the _transcribe_channel method
        result = calculator._transcribe_channel("mock_path.mp3", "test_speaker")
        print("\nTest _transcribe_channel result:")
        print(json.dumps(result, indent=2))

        # Verify result
        assert len(result) == 2, "Expected 2 words, spaces should be filtered out"
        assert result[0]["text"] == "Hello", "First word should be 'Hello'"
        assert result[0]["speaker"] == "test_speaker", "Speaker label should be set"
        assert "logprob" in result[0], "Additional fields should be preserved"

        print("✓ _transcribe_channel test passed!")
        return result

    except Exception as e:
        print(f"× Test failed: {str(e)}")
        raise
    finally:
        # Restore original method if it exists
        if calculator.elevenlabs_client:
            calculator.elevenlabs_client.speech_to_text.convert = original_method


def test_merge_and_format_transcript(words_customer=None, words_agent=None):
    """Test the _merge_and_format_transcript method"""
    calculator = AudioMetricsCalculator()

    # Create test word data if not provided
    if words_customer is None:
        words_customer = [
            {"text": "Hello", "start": 0.0, "end": 0.5, "speaker": "customer"},
            {"text": "there", "start": 0.6, "end": 0.9, "speaker": "customer"},
        ]

    if words_agent is None:
        words_agent = [
            {
                "text": "Hi",
                "start": 0.8,
                "end": 1.0,
                "speaker": "ai_agent",
            },  # Overlaps with "there"
            {"text": "how", "start": 1.1, "end": 1.3, "speaker": "ai_agent"},
            {"text": "are", "start": 1.4, "end": 1.6, "speaker": "ai_agent"},
            {"text": "you", "start": 1.7, "end": 2.0, "speaker": "ai_agent"},
        ]

    # Test the merge and format method
    result = calculator._merge_and_format_transcript(words_customer, words_agent)
    print("\nTest _merge_and_format_transcript result:")
    print(json.dumps(result, indent=2))

    # Verify result
    assert "words" in result, "Result should contain 'words'"
    assert "dialog" in result, "Result should contain 'dialog'"
    assert "has_overlaps" in result, "Result should contain 'has_overlaps'"
    assert "overlap_count" in result, "Result should contain 'overlap_count'"

    print(f"Detected {result['overlap_count']} overlapping words")
    print("✓ _merge_and_format_transcript test passed!")
    return result


def test_full_pipeline():
    """Test the full pipeline with a sample file"""
    calculator = AudioMetricsCalculator()

    # Get a list of test files
    test_files = os.listdir(calculator.input_dir)
    if not test_files:
        print(
            "No test files found. Please add files to the stereo_test_calls directory."
        )
        return

    # Process the first file
    test_file = test_files[0]
    print(f"\nTesting full pipeline with file: {test_file}")

    # Process the file
    metrics = calculator.process_file(test_file)

    # Check if transcript data exists
    if metrics.get("transcript_data"):
        print(
            f"Transcript generated successfully with {len(metrics['transcript_data']['words'])} words"
        )
        if metrics["transcript_data"].get("has_overlaps"):
            print(
                f"Detected {metrics['transcript_data'].get('overlap_count', 0)} overlapping words"
            )
        print("Sample dialog:")
        print(
            metrics["transcript_data"]["dialog"][:200] + "..."
        )  # Show first 200 chars
        print("✓ Full pipeline test passed!")
    else:
        print(
            "No transcript data generated. Check if ElevenLabs API key is configured."
        )


def test_database_storage():
    """Test storing and retrieving transcript data from the database"""
    calculator = AudioMetricsCalculator()
    db_session = next(get_db())

    try:
        # Get a list of test files
        test_files = os.listdir(calculator.input_dir)
        if not test_files:
            print(
                "No test files found. Please add files to the stereo_test_calls directory."
            )
            return

        # Use the first file
        test_file = test_files[0]
        print(f"\nTesting database storage with file: {test_file}")

        # First process the file
        calculator.process_file(test_file)

        # Then retrieve it from the database
        analysis = get_analysis_by_filename(db_session, test_file)
        if not analysis:
            print("× Database test failed: Analysis not found in database")
            return

        # Recreate metrics from database
        metrics = recreate_metrics_from_db(analysis)

        # Check if transcript data exists and has the expected fields
        if not metrics.get("transcript_data"):
            print(
                "× Database test failed: Transcript data not found in retrieved metrics"
            )
            return

        transcript_data = metrics["transcript_data"]
        assert "dialog" in transcript_data, "Transcript should contain dialog"
        assert "words" in transcript_data, "Transcript should contain words"
        assert "has_overlaps" in transcript_data, (
            "Transcript should contain has_overlaps"
        )
        assert "overlap_count" in transcript_data, (
            "Transcript should contain overlap_count"
        )

        print("✓ Database storage test passed!")
        print(
            f"Successfully retrieved transcript with {len(transcript_data['words'])} words "
            + f"and {transcript_data.get('overlap_count', 0)} overlaps"
        )

    finally:
        db_session.close()


if __name__ == "__main__":
    # Check if ElevenLabs API key is configured
    calculator = AudioMetricsCalculator()
    if not calculator.elevenlabs_client:
        print("WARNING: ELEVENLABS_API_KEY not configured. Some tests may be skipped.")

    print("Running transcript handling tests...")

    # Run the tests (uncomment as needed)
    if calculator.elevenlabs_client:
        customer_words = test_transcribe_channel()
        agent_words = test_transcribe_channel()
        test_merge_and_format_transcript(customer_words, agent_words)
    else:
        # Run merge test with mock data
        test_merge_and_format_transcript()

    # Test the full pipeline
    test_full_pipeline()

    # Test database storage
    test_database_storage()

    print("\nAll tests completed!")
