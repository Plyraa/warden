#!/usr/bin/env python3
"""
Comprehensive Test Suite for Warden API

This script combines:
- Batch client functionality (from batch_client_example.py)
- New format testing (from test_new_format.py)
- Error handling testing (from test_error_handling.py)

Usage:
    python comprehensive_test.py
"""

import json
import os
import requests


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
                "words_per_minute": 120.0,
            },
            {"status": "file_not_found"},
            {"status": "download_error"},
        ]
    }

    results = simulated_response["results"]

    # Count successes and errors
    success_count = len([r for r in results if r.get("status") == "success"])
    error_count = len([r for r in results if r.get("status") != "success"])

    print(f"Total results: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    print("\n--- Successful Results ---")
    for result in results:
        if result.get("status") == "success":
            print(
                f"✅ {result['filename']}: {result['average_latency']:.1f}ms avg latency"
            )
            print(f"   Status: {result['status']}")
            print(
                f"   Metrics: p50={result['p50_latency']:.1f}ms, p90={result['p90_latency']:.1f}ms"
            )

    print("\n--- Error Results ---")
    for result in results:
        if result.get("status") != "success":
            print(f"❌ Status: {result['status']}")


def test_api_health_and_endpoints(base_url="http://127.0.0.1:8000"):
    """Test API health and basic endpoints"""
    print("=" * 60)
    print("TEST 2: API HEALTH AND ENDPOINTS")
    print("=" * 60)

    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")
        return False

    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✓ Root endpoint working")
            root_data = response.json()
            print(f"   Message: {root_data.get('message', 'N/A')}")
            print(f"   Endpoints: {root_data.get('endpoints', 'N/A')}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")

    return True


def test_file_validation(base_url="http://127.0.0.1:8000"):
    """Test file validation endpoint"""
    print("=" * 60)
    print("TEST 3: FILE VALIDATION")
    print("=" * 60)

    print("Testing file validation...")
    test_files = [
        "stereo_test_calls/test1.mp3",  # Should exist
        "nonexistent_file.mp3",  # Should not exist
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3",  # URL
        "https://invalid-domain-12345.com/fake.mp3",  # Invalid URL
    ]

    try:
        response = requests.post(
            f"{base_url}/validate", json={"file_paths": test_files}
        )
        if response.status_code == 200:
            print("✓ Validation endpoint working")
            validation_data = response.json()
            for file_info in validation_data["files"]:
                status = "✓" if file_info["accessible"] else "✗"
                print(f"   {status} {file_info['file_path']}")
                if file_info.get("error"):
                    print(f"      Error: {file_info['error']}")
                if file_info.get("note"):
                    print(f"      Note: {file_info['note']}")
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
        "stereo_test_calls/test1.mp3",  # Should work if file exists
        "definitely_does_not_exist.mp3",  # File not found error
        "https://invalid-url-test.com/fake.mp3",  # Download error
    ]

    try:
        response = requests.post(f"{base_url}/batch", json={"file_paths": batch_files})
        if response.status_code == 200:
            print("✓ Batch processing completed")
            batch_data = response.json()

            # Count successes and errors from results
            results = batch_data["results"]
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
    """Test the complete batch workflow with real files"""
    print("=" * 60)
    print("TEST 5: FULL BATCH WORKFLOW")
    print("=" * 60)

    base_url = "http://127.0.0.1:8000"

    # Check if API is running
    if not check_api_health(base_url):
        print(
            "❌ API server is not running. Please start it with: python warden.py --api-only"
        )
        return

    print("Choose test scenario:")
    print("1: Test with local files only")
    print("2: Test with URLs only")
    print("3: Test with mixed local files and URLs")
    print("4: Run automatic test with available files")

    try:
        choice = input("Enter your choice (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nRunning automatic test...")
        choice = "4"

    file_paths = []

    if choice == "1":
        # Local files only
        base_dir = "stereo_test_calls"
        if os.path.exists(base_dir):
            files = [f for f in os.listdir(base_dir) if f.endswith((".mp3", ".wav"))]
            if files:
                sample_files = files[:2]  # Use first 2 files
                file_paths = [
                    os.path.abspath(os.path.join(base_dir, f)) for f in sample_files
                ]
                print(f"\nFound {len(file_paths)} local files:")
                for path in file_paths:
                    print(f" - {path}")
            else:
                print("No audio files found in stereo_test_calls directory")
                return
        else:
            print("stereo_test_calls directory not found")
            return

    elif choice == "2":
        # URLs only
        file_paths = [
            "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/243801406824559add7684.37683750.mp3",
            "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/16952962156823ffe2a06954.22932860.mp3",
        ]
        print(f"\nUsing {len(file_paths)} URL references:")
        for url in file_paths:
            print(f" - {url}")

    elif choice == "3":
        # Mixed local and URLs
        base_dir = "stereo_test_calls"
        if os.path.exists(base_dir):
            files = [f for f in os.listdir(base_dir) if f.endswith((".mp3", ".wav"))]
            if files:
                local_file = os.path.abspath(os.path.join(base_dir, files[0]))
                file_paths.append(local_file)

        file_paths.append(
            "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3"
        )

        print(f"\nUsing {len(file_paths)} mixed files:")
        for path in file_paths:
            print(f" - {path}")

    else:  # choice == "4" or default
        # Automatic test with error handling demo
        print("\nRunning automatic test with mixed scenarios...")
        base_dir = "stereo_test_calls"
        if os.path.exists(base_dir):
            files = [f for f in os.listdir(base_dir) if f.endswith((".mp3", ".wav"))]
            if files:
                file_paths.append(os.path.join(base_dir, files[0]))

        # Add some URLs and intentional errors
        file_paths.extend(
            [
                "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3",
                "nonexistent_file.mp3",  # This will cause an error
            ]
        )

        print(f"Testing with {len(file_paths)} files:")
        for path in file_paths:
            print(f" - {path}")

    if not file_paths:
        print("No files to process")
        return

    # Optional: Validate files first
    print("\n--- VALIDATION PHASE ---")
    validation_results = validate_files(file_paths, base_url)
    if validation_results:
        for file_info in validation_results["files"]:
            status = "✓" if file_info["accessible"] else "✗"
            print(f"{status} {file_info['file_path']}")
            if file_info.get("error"):
                print(f"   Error: {file_info['error']}")
            if file_info.get("note"):
                print(f"   Note: {file_info['note']}")

    # Send batch request
    print("\n--- BATCH PROCESSING PHASE ---")
    response = analyze_files(file_paths, base_url)

    if response:
        print("\n--- BATCH RESULTS ---")

        # Count successes and errors
        results = response["results"]
        success_count = len([r for r in results if r.get("status") == "success"])
        error_count = len([r for r in results if r.get("status") != "success"])

        print(f"Success Count: {success_count}")
        print(f"Error Count: {error_count}")
        print(f"Total Files Processed: {len(results)}")

        # Display successful results
        successful_results = [r for r in results if r.get("status") == "success"]
        if successful_results:
            print(f"\n--- SUCCESSFUL ANALYSES ({success_count}) ---")
            for i, result in enumerate(successful_results):
                print(f"\n=== File {i + 1}: {result['filename']} ===")

                # Display latency metrics
                print("\nLatency Metrics:")
                print(f"Average Latency: {result['average_latency']:.2f} ms")
                print(f"P50 Latency: {result['p50_latency']:.2f} ms")
                print(f"P90 Latency: {result['p90_latency']:.2f} ms")
                print(f"Min Latency: {result['min_latency']:.2f} ms")
                print(f"Max Latency: {result['max_latency']:.2f} ms")

                # Display interruption metrics
                print("\nInterruptions:")
                print(
                    f"AI Interrupting User: {'Yes' if result['ai_interrupting_user'] else 'No'}"
                )
                print(
                    f"User Interrupting AI: {'Yes' if result['user_interrupting_ai'] else 'No'}"
                )

                # Display overlap counts
                print("\nOverlaps:")
                print(f"AI→User Overlaps: {result['ai_user_overlap_count']}")
                print(f"User→AI Overlaps: {result['user_ai_overlap_count']}")

                # Display other metrics
                print("\nOther Metrics:")
                print(f"Talk Ratio: {result['talk_ratio']:.2f}")
                print(f"Average Pitch: {result['average_pitch']:.2f} Hz")
                print(f"Words Per Minute: {result['words_per_minute']:.2f}")

                # Display latency points (first 5 only)
                if result["latency_points"]:
                    print("\nLatency Points (first 5):")
                    for point in result["latency_points"][:5]:
                        print(
                            f" - {point['latency_ms']:.2f} ms at {point['moment']:.2f}s"
                        )

                    if len(result["latency_points"]) > 5:
                        print(
                            f" ... and {len(result['latency_points']) - 5} more points"
                        )

                # Pretty print the complete metric object
                print("\n=== Complete Metric Object ===")
                print(json.dumps(result, indent=2, sort_keys=True))
                print("-" * 50)

                print("-" * 60)

        # Display errors
        error_results = [r for r in results if r.get("status") != "success"]
        if error_results:
            print(f"\n--- ERRORS ({error_count}) ---")
            for i, error in enumerate(error_results):
                print(f"\n=== Error {i + 1} ===")
                print(f"Status: {error['status']}")
                print("-" * 40)

        # Error handling demo
        print("\n--- ERROR HANDLING DEMO ---")
        print("Testing error handling with invalid files...")

        test_files = [
            "nonexistent_file.mp3",  # File not found
            "https://invalid-url-that-does-not-exist.com/fake.mp3",  # Download error
        ]

        test_response = analyze_files(test_files, base_url)
        if test_response:
            test_results = test_response["results"]
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
