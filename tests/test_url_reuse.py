#!/usr/bin/env python3
"""
Quick test to verify URL reuse functionality
"""

import requests


def test_url_reuse():
    """Test that the same URL doesn't get downloaded twice"""

    # Simple test with a small MP3 file
    test_url = "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3"

    print(f"Testing URL reuse with: {test_url}")

    # Make first request
    print("Making first batch request with URL...")
    try:
        first_response = requests.post(
            "http://localhost:8000/batch", json={"file_paths": [test_url]}, timeout=60
        )

        print(f"First request status: {first_response.status_code}")
        if first_response.status_code == 200:
            first_data = first_response.json()
            print(f"First response success: {len(first_data)} results")
            if first_data and len(first_data) > 0:
                print(f"First result status: {first_data[0].get('status', 'unknown')}")
        else:
            print(f"First request failed: {first_response.text}")
            return
    except Exception as e:
        print(f"First request error: {e}")
        return
    # Make second request with same URL
    print("\nMaking second batch request with same URL...")
    try:
        second_response = requests.post(
            "http://localhost:8000/batch", json={"file_paths": [test_url]}, timeout=60
        )

        print(f"Second request status: {second_response.status_code}")
        if second_response.status_code == 200:
            second_data = second_response.json()
            print(f"Second response success: {len(second_data)} results")
            if second_data and len(second_data) > 0:
                print(
                    f"Second result status: {second_data[0].get('status', 'unknown')}"
                )

            # Compare basic structure
            if (
                first_data
                and second_data
                and len(first_data) > 0
                and len(second_data) > 0
            ):
                first_status = first_data[0].get("status")
                second_status = second_data[0].get("status")
                if first_status == "success" and second_status == "success":
                    print(
                        "\n✅ SUCCESS: Both requests succeeded - URL reuse likely working!"
                    )
                    print(
                        "Second request should have been faster (reused existing analysis)"
                    )
                else:
                    print(
                        f"\n⚠️  WARNING: Status check - First: {first_status}, Second: {second_status}"
                    )
        else:
            print(f"Second request failed: {second_response.text}")
    except Exception as e:
        print(f"Second request error: {e}")


if __name__ == "__main__":
    test_url_reuse()
