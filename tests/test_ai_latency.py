#!/usr/bin/env python3
"""
Test script for the improved turn-taking latency calculation and visualization.
This verifies that the new overlap-aware algorithm works correctly and focuses on AI responses.
"""

from audio_metrics import AudioMetricsCalculator


def test_ai_response_latency():
    """Test the improved latency calculation focusing on AI agent responses."""
    calc = AudioMetricsCalculator()

    print("Testing AI Response Latency Calculation")
    print("=" * 50)

    # Sample conversation data
    user_segments = [
        {"start": 13.6, "end": 17.6},  # User speaks
        {"start": 46.5, "end": 48.3},  # User speaks again
        {"start": 70.8, "end": 71.4},  # Short user comment
        {"start": 85.8, "end": 89.6},  # User speaks
        {"start": 104.1, "end": 110.5},  # User speaks
        {"start": 124.3, "end": 151.6},  # Long user turn
    ]

    agent_segments = [
        {"start": 0.6, "end": 5.6},  # AI opens conversation
        {
            "start": 21.2,
            "end": 43.6,
        },  # AI responds to first user (latency: 21.2 - 17.6 = 3.6s)
        {
            "start": 52.3,
            "end": 69.3,
        },  # AI responds to second user (latency: 52.3 - 48.3 = 4.0s)
        {
            "start": 76.0,
            "end": 84.1,
        },  # AI responds to short comment (latency: 76.0 - 71.4 = 4.6s)
        {
            "start": 92.8,
            "end": 103.0,
        },  # AI responds to fourth user (latency: 92.8 - 89.6 = 3.2s)
        {
            "start": 116.0,
            "end": 122.9,
        },  # AI responds to fifth user (latency: 116.0 - 110.5 = 5.5s)
    ]

    # Calculate latencies
    stats, details = calc.calculate_turn_taking_latency(user_segments, agent_segments)

    print(f"Overall Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}s")
        else:
            print(f"  {key}: {value}")

    # Filter for AI responses only (what the visualization will show)
    ai_responses = [
        d for d in details if d.get("interaction_type", "").startswith("user_to_agent")
    ]

    print(f"\nAI Agent Response Analysis:")
    print(f"Total conversation turns: {len(details)}")
    print(f"AI responses to user: {len(ai_responses)}")

    if ai_responses:
        ai_latencies = [d["latency_seconds"] for d in ai_responses]
        print(f"\nAI Response Latencies:")
        print(f"  Average: {sum(ai_latencies) / len(ai_latencies):.3f}s")
        print(f"  Min: {min(ai_latencies):.3f}s")
        print(f"  Max: {max(ai_latencies):.3f}s")

        print(f"\nDetailed AI Responses (what visualization will show):")
        for i, response in enumerate(ai_responses, 1):
            print(f"  AI Response {i}:")
            print(f"    Time: {response['to_turn_start']:.1f}s")
            print(f"    Latency: {response['latency_seconds']:.3f}s")
            print(f"    Rating: {response['rating']}")
            print(f"    Previous user turn ended: {response['from_turn_end']:.1f}s")

    return ai_responses


def test_visualization_data():
    """Test that visualization will receive the correct data format."""
    print("\n" + "=" * 50)
    print("Testing Visualization Data Format")
    print("=" * 50)

    ai_responses = test_ai_response_latency()

    if ai_responses:
        print(f"\nData that visualization will receive:")
        print(f"Fields available in each latency detail:")
        sample = ai_responses[0]
        for key in sorted(sample.keys()):
            print(f"  {key}: {sample[key]}")

        print(f"\nVisualization will filter for items where:")
        print(f"  interaction_type.startswith('user_to_agent') = True")
        print(f"  This gives us {len(ai_responses)} data points to plot")

        print(f"\nPlot data:")
        print(f"  X-axis (time): [to_turn_start values]")
        print(f"  Y-axis (latency): [latency_seconds values]")
        print(f"  Colors: Based on [rating values]")
        print(f"  Labels: AI #1, AI #2, etc.")


if __name__ == "__main__":
    test_visualization_data()
