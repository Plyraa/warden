#!/usr/bin/env python3
"""
Test script for the improved calculate_turn_taking_latency function
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_metrics import AudioMetricsCalculator


def test_improved_latency_calculation():
    """Test the improved latency calculation with the provided example data"""

    # Create an instance of the calculator
    calculator = AudioMetricsCalculator()

    # Test data based on the user's example - simulating the merged turns they provided
    # These represent the final merged turns that were causing issues
    user_segments = [
        {"start": 13.6, "end": 17.6},  # User turn 1
        {"start": 46.5, "end": 48.3},  # User turn 2
        {"start": 70.8, "end": 71.4},  # User turn 3
        {"start": 85.8, "end": 89.6},  # User turn 4
        {"start": 104.1, "end": 110.5},  # User turn 5
        {"start": 124.3, "end": 151.6},  # User turn 6
    ]

    agent_segments = [
        {"start": 0.6, "end": 5.6},  # AI turn 1
        {"start": 21.2, "end": 43.6},  # AI turn 2
        {"start": 52.3, "end": 69.3},  # AI turn 3
        {"start": 76.0, "end": 84.1},  # AI turn 4
        {"start": 92.8, "end": 103.0},  # AI turn 5
        {"start": 116.0, "end": 122.9},  # AI turn 6
    ]

    print("=== Testing Improved Turn-Taking Latency Calculation ===")
    print("\nInput segments:")
    print("User segments:")
    for i, seg in enumerate(user_segments):
        print(f"  {i + 1}: {seg['start']:.1f}s - {seg['end']:.1f}s")
    print("Agent segments:")
    for i, seg in enumerate(agent_segments):
        print(f"  {i + 1}: {seg['start']:.1f}s - {seg['end']:.1f}s")

    # Test the improved function
    latency_stats, latency_details = calculator.calculate_turn_taking_latency(
        user_segments, agent_segments
    )

    print("\n=== Results ===")
    print("Latency Statistics:")
    for key, value in latency_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}s")
        else:
            print(f"  {key}: {value}")

    print(f"\nDetailed Latencies ({len(latency_details)} transitions):")
    for i, detail in enumerate(latency_details):
        interaction = detail.get("interaction_type", "unknown")
        from_speaker = detail.get("from_speaker", "unknown")
        to_speaker = detail.get("to_speaker", "unknown")
        latency = detail["latency_seconds"]
        rating = detail.get("rating", "unknown")

        print(f"  {i + 1}: {from_speaker} -> {to_speaker}")
        print(f"      Latency: {latency:.3f}s ({latency * 1000:.0f}ms) - {rating}")
        print(f"      Type: {interaction}")
        print()


def test_overlap_scenario():
    """Test with overlapping segments to verify overlap handling"""

    calculator = AudioMetricsCalculator()

    print("\n=== Testing Overlap Handling ===")

    # Create scenario with overlaps
    user_segments = [
        {"start": 1.0, "end": 5.0},  # User speaks first
        {"start": 8.0, "end": 12.0},  # User responds
        {"start": 15.0, "end": 20.0},  # User interrupts AI (overlap)
    ]

    agent_segments = [
        {"start": 6.0, "end": 10.0},  # AI responds (overlaps with user)
        {"start": 13.0, "end": 18.0},  # AI speaks (gets interrupted)
        {"start": 22.0, "end": 25.0},  # AI continues after interruption
    ]

    print("Overlap test segments:")
    print("User segments:")
    for i, seg in enumerate(user_segments):
        print(f"  {i + 1}: {seg['start']:.1f}s - {seg['end']:.1f}s")
    print("Agent segments:")
    for i, seg in enumerate(agent_segments):
        print(f"  {i + 1}: {seg['start']:.1f}s - {seg['end']:.1f}s")

    latency_stats, latency_details = calculator.calculate_turn_taking_latency(
        user_segments, agent_segments
    )

    print("\nOverlap Test Results:")
    print("Latency Statistics:")
    for key, value in latency_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}s")
        else:
            print(f"  {key}: {value}")

    print(f"\nDetailed Latencies ({len(latency_details)} transitions):")
    for i, detail in enumerate(latency_details):
        interaction = detail.get("interaction_type", "unknown")
        from_speaker = detail.get("from_speaker", "unknown")
        to_speaker = detail.get("to_speaker", "unknown")
        latency = detail["latency_seconds"]
        rating = detail.get("rating", "unknown")

        print(f"  {i + 1}: {from_speaker} -> {to_speaker}")
        print(f"      Latency: {latency:.3f}s ({latency * 1000:.0f}ms) - {rating}")
        print(f"      Type: {interaction}")


if __name__ == "__main__":
    try:
        test_improved_latency_calculation()
        test_overlap_scenario()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
