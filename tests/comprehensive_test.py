#!/usr/bin/env python3
"""
Comprehensive Test Suite for Warden API

This script combines:
- Batch client functionality (from batch_client_example.py)
- New format testing (from test_new_format.py)
- Error handling testing (from test_error_handling.py)

Usage:
    python tests/comprehensive_test.py
"""

import json
import os
import requests

# Invoke-RestMethod -Uri "http://127.0.0.1:8000/batch" -Method POST -Body '{"file_paths": ["https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/66057328368247d623d0b87.67876133.mp3"]}' -ContentType "application/json" | ConvertTo-Json -Depth 10
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def download_file(url, save_path):
    """Download a file from URL to local path"""
    print(f"Downloading {url} to {save_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded file to {save_path}")
        return save_path

    except Exception as e:
        print(f"Error downloading file from {url}: {str(e)}")
        return None


def check_api_health(base_url="http://127.0.0.1:8000"):
    """Check if the API server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"API not reachable: {e}")
        return False


# =============================================================================
# API INTERACTION FUNCTIONS
# =============================================================================


def analyze_files(file_paths, base_url="http://127.0.0.1:8000"):
    """
    Send a batch request to the Warden API to analyze audio files

    Args:
        file_paths: List of paths to audio files or URLs to analyze
        base_url: Base URL of the API server

    Returns:
        The response object containing results, or None if request failed
    """
    url = f"{base_url}/batch"
    data = {"file_paths": file_paths}

    print(f"\nSending batch request to {url}...")
    print(f"Processing {len(file_paths)} files/URLs")

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        if hasattr(e, "response") and e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def validate_files(file_paths, base_url="http://127.0.0.1:8000"):
    """
    Validate files before processing them

    Args:
        file_paths: List of paths to audio files or URLs to validate
        base_url: Base URL of the API server

    Returns:
        The validation results or None if request failed
    """
    url = f"{base_url}/validate"
    data = {"file_paths": file_paths}

    print(f"\nValidating {len(file_paths)} files/URLs...")

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Validation error: {str(e)}")
        return None


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_new_format_simulation():
    """Test the new response format with simulated data"""
    print("=" * 60)
    print("TEST 1: NEW FORMAT SIMULATION")
    print("=" * 60)

    # Simulate API response with new format
    simulated_response = {
        "results": [
            {
                "filename": "test1.mp3",
                "status": "success",
                "average_latency": 1250.5,
                "p50_latency": 1100.0,
                "p90_latency": 2000.0,
                "min_latency": 800.0,
                "max_latency": 2100.0,
                "latency_points": [
                    {"latency_ms": 1200, "moment": 5.2},
                    {"latency_ms": 1100, "moment": 8.7},
                ],
                "ai_interrupting_user": False,
                "user_interrupting_ai": True,
                "ai_user_overlap_count": 0,
                "user_ai_overlap_count": 2,
                "talk_ratio": 0.65,
                "average_pitch": 180.5,
                "words_per_minute": 150.2,
            },
            {
                "filename": "test2.mp3",
                "status": "success",
                "average_latency": 980.2,
                "p50_latency": 900.0,
                "p90_latency": 1500.0,
                "min_latency": 700.0,
                "max_latency": 1600.0,
                "latency_points": [
                    {"latency_ms": 900, "moment": 3.1},
                    {"latency_ms": 1050, "moment": 12.4},
                ],
                "ai_interrupting_user": True,
                "user_interrupting_ai": False,
                "ai_user_overlap_count": 1,
                "user_ai_overlap_count": 0,
                "talk_ratio": 1.2,
                "average_pitch": 165.3,
                "words_per_minute": 142.8,
            },
        ]
    }

    print("Simulated API response (new format):")
    print(json.dumps(simulated_response, indent=2))

    # Process simulation data
    results = simulated_response["results"]
    print(f"\n📊 Analysis Results Summary:")
    print(f"Total files processed: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['filename']}:")
        print(f"   Status: {result['status']}")
        if result["status"] == "success":
            print(f"   Average Latency: {result['average_latency']:.1f}ms")
            print(f"   Talk Ratio: {result['talk_ratio']:.2f}")
            print(f"   WPM: {result['words_per_minute']:.1f}")
            print(f"   Pitch: {result['average_pitch']:.1f}Hz")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")

    print("✅ Format simulation test completed")


def test_api_health_and_endpoints(base_url="http://127.0.0.1:8000"):
    """Test basic API health and endpoint availability"""
    print("=" * 60)
    print("TEST 2: API HEALTH AND ENDPOINTS")
    print("=" * 60)

    # Test health endpoint
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")

    # Test batch endpoint with minimal payload
    print("\nTesting /batch endpoint with empty payload...")
    try:
        response = requests.post(f"{base_url}/batch", json={"file_paths": []})
        print(f"✓ Batch endpoint responds (status: {response.status_code})")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"✗ Batch endpoint error: {e}")


def test_file_validation(base_url="http://127.0.0.1:8000"):
    """Test file validation functionality"""
    print("=" * 60)
    print("TEST 3: FILE VALIDATION")
    print("=" * 60)

    test_files = [
        "../stereo_test_calls/test1.mp3",  # Should exist
        "nonexistent_file.mp3",  # Should not exist
        # "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3",  # URL
    ]

    print("Testing file validation...")
    try:
        response = requests.post(f"{base_url}/validate", json={"file_paths": test_files})
        if response.status_code == 200:
            print("✓ Validation endpoint working")
            validation_data = response.json()
            print(f"   Response: {json.dumps(validation_data, indent=2)}")
        else:
            print(f"✗ Validation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"✗ Validation error: {e}")


def test_batch_processing_with_errors(base_url="http://127.0.0.1:8000"):
    """Test batch processing with intentional errors"""
    print("=" * 60)
    print("TEST 4: BATCH PROCESSING WITH ERRORS")
    print("=" * 60)

    print("Testing batch processing with mixed valid/invalid files...")
    batch_files = [
        "../stereo_test_calls/test1.mp3",  # Should work if file exists
        "definitely_does_not_exist.mp3",  # File not found error
        "https://invalid-url-test.com/fake.mp3",  # Download error
    ]

    try:
        response = requests.post(f"{base_url}/batch", json={"file_paths": batch_files})
        if response.status_code == 200:
            print("✓ Batch processing completed")
            batch_data = response.json()  # Count successes and errors from results
            results = batch_data
            success_count = len([r for r in results if r.get("status") == "success"])
            error_count = len([r for r in results if r.get("status") != "success"])

            print(f"   Success count: {success_count}")
            print(f"   Error count: {error_count}")

            # Show successful results
            successful_results = [r for r in results if r.get("status") == "success"]
            if successful_results:
                print("\n   Successful analyses:")
                for i, result in enumerate(successful_results):
                    print(
                        f"     {i + 1}. {result['filename']} - {result['average_latency']:.2f}ms avg latency"
                    )

            # Show errors
            error_results = [r for r in results if r.get("status") != "success"]
            if error_results:
                print("\n   Errors encountered:")
                for i, error in enumerate(error_results):
                    print(f"     {i + 1}. Status: {error['status']}")
        else:
            print(f"✗ Batch processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"✗ Batch processing error: {e}")


def test_full_batch_workflow():
    """Test the complete batch processing workflow"""
    print("=" * 60)
    print("TEST 5: FULL BATCH WORKFLOW")
    print("=" * 60)

    base_url = "http://127.0.0.1:8000"

    print("Step 1: Check for local test files")
    local_files = []
    try:
        base_dir = "../stereo_test_calls"
        if os.path.exists(base_dir):
            audio_files = [
                f
                for f in os.listdir(base_dir)
                if f.endswith((".mp3", ".wav")) and not f.endswith("_downsampled.mp3")
            ]
            if audio_files:
                local_files = [f"{base_dir}/{audio_files[0]}"]  # Just use first file
                print(f"Found local test files: {local_files}")
            else:
                print("No audio files found in ../stereo_test_calls directory")
        else:
            print("../stereo_test_calls directory not found")
    except Exception as e:
        print(f"Error checking local files: {e}")

    # Fallback to URLs
    url_files = [
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/243801406824559add7684.37683750.mp3",
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/16952962156823ffe2a06954.22932860.mp3",
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3",
    ]

    # Use local files if available, otherwise URLs
    test_files = local_files if local_files else url_files[:2]  # Limit to 2 for speed

    print(f"\nStep 2: Process files with Warden API")
    print(f"Files to process: {test_files}")

    if local_files:
        base_dir = "../stereo_test_calls"
        if os.path.exists(base_dir):
            available_files = [
                f
                for f in os.listdir(base_dir)
                if f.endswith((".mp3", ".wav")) and not f.endswith("_downsampled.mp3")
            ]
            test_files = [f"{base_dir}/{f}" for f in available_files[:2]]  # Use first 2

        print("Using file URLs for consistent testing...")
        test_files = [
            "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3"
        ]

    # Enhanced batch processing test
    print("Step 3: Enhanced Batch Processing Analysis")
    enhanced_test_files = []
    try:
        base_dir = "../stereo_test_calls"
        if os.path.exists(base_dir):
            audio_files = [
                f
                for f in os.listdir(base_dir)
                if f.endswith((".mp3", ".wav")) and not f.endswith("_downsampled.mp3")
            ]
            if audio_files:
                enhanced_test_files = [
                    f"{base_dir}/{audio_files[0]}",  # Local file
                    "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3",
                ][:2]  # Limit for demo
    except Exception as e:
        print(f"Error preparing enhanced test: {e}")

    if not enhanced_test_files:
        enhanced_test_files = [
            "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3"
        ]

    print(f"Enhanced test files: {enhanced_test_files}")

    # Process files
    response = analyze_files(enhanced_test_files, base_url)
    if response:
        print("\n🎯 BATCH PROCESSING RESULTS:")
        print("-" * 40)

        results = response
        success_count = len([r for r in results if r.get("status") == "success"])
        error_count = len([r for r in results if r.get("status") != "success"])

        print(f"✅ Successfully processed: {success_count} files")
        print(f"❌ Errors encountered: {error_count} files")

        print("\n📈 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. File: {result.get('filename', 'Unknown')}")
            print(f"   Status: {result.get('status', 'Unknown')}")

            if result.get("status") == "success":
                # Show key metrics
                avg_lat = result.get("average_latency", 0)
                talk_ratio = result.get("talk_ratio", 0)
                wpm = result.get("words_per_minute", 0)
                pitch = result.get("average_pitch", 0)

                print(f"   📊 Average Latency: {avg_lat:.1f}ms")
                print(f"   💬 Talk Ratio: {talk_ratio:.2f}")
                print(f"   🗣️  Words/Minute: {wpm:.1f}")
                print(f"   🎵 Average Pitch: {pitch:.1f}Hz")

                # Show overlap info
                ai_interrupts = result.get("ai_user_overlap_count", 0)
                user_interrupts = result.get("user_ai_overlap_count", 0)
                print(f"   🔄 AI Interruptions: {ai_interrupts}")
                print(f"   🔄 User Interruptions: {user_interrupts}")

                # Show latency range
                min_lat = result.get("min_latency", 0)
                max_lat = result.get("max_latency", 0)
                p50_lat = result.get("p50_latency", 0)
                p90_lat = result.get("p90_latency", 0)
                print(f"   📏 Latency Range: {min_lat:.0f}-{max_lat:.0f}ms")
                print(f"   📊 P50/P90: {p50_lat:.0f}/{p90_lat:.0f}ms")

        print("\n🔧 ERROR HANDLING DEMO:")
        print("Testing error scenarios...")

        test_files = [
            "nonexistent_file.mp3",  # File not found
            "https://invalid-url-that-does-not-exist.com/fake.mp3",  # Download error
        ]

        test_response = analyze_files(test_files, base_url)
        if test_response:
            test_results = test_response
            test_success = len(
                [r for r in test_results if r.get("status") == "success"]
            )
            test_errors = len([r for r in test_results if r.get("status") != "success"])
            print(f"Demo: {test_success} successful, {test_errors} errors")
            for result in test_results:
                if result.get("status") != "success":
                    print(f"  {result['status']}")
    else:
        print("Failed to get results from API")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Main function to run all tests"""
    print("🔧 WARDEN API COMPREHENSIVE TEST SUITE 🔧")
    print("=" * 60)

    # Test 1: Format simulation (doesn't require API)
    test_new_format_simulation()

    # Test 2-5: API tests (require running API server)
    base_url = "http://127.0.0.1:8000"

    if check_api_health(base_url):
        print(f"\n✅ API server is running at {base_url}")

        test_api_health_and_endpoints(base_url)
        test_file_validation(base_url)
        test_batch_processing_with_errors(base_url)
        test_full_batch_workflow()

    else:
        print(f"\n❌ API server is not running at {base_url}")
        print("To run API tests, start the server with: python warden.py --api-only")
        print("Then run this test script again.")

    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE TEST SUITE COMPLETE 🎉")


if __name__ == "__main__":
    main()
