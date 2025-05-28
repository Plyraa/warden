import requests
import os
import time
from urllib.parse import urlparse

# Example script to demonstrate using the Warden batch API endpoint


def download_mp3_from_url(url, save_dir):
    """
    Download an MP3 file from a URL and save it to a directory

    Args:
        url: URL of the MP3 file
        save_dir: Directory to save the downloaded file

    Returns:
        Path to the downloaded file or None if download failed
    """
    try:
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # If filename is empty or doesn't end with .mp3, create a unique name
        if not filename or not filename.lower().endswith(".mp3"):
            filename = f"downloaded_{int(time.time())}.mp3"

        # Create the save path
        save_path = os.path.join(save_dir, filename)

        # Download the file
        print(f"Downloading {url} to {save_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded file to {save_path}")
        return save_path

    except Exception as e:
        print(f"Error downloading file from {url}: {str(e)}")
        return None


def analyze_files(file_paths):
    """
    Send a batch request to the Warden API to analyze audio files

    Args:
        file_paths: List of paths to audio files or URLs to analyze
                   The server can handle both local file paths and URLs

    Returns:
        The metrics for each file
    """
    url = "http://127.0.0.1:8000/batch"
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


def main():
    # Let user choose to use URLs or local files
    print("Warden Batch Client Example")
    print("--------------------------")
    print("1: Analyze local files")
    print("2: Analyze files from URLs")
    print("3: Analyze both local files and URLs")

    choice = input("Enter your choice (1-3): ")

    file_paths = []

    # Process local files
    if choice in ["1", "3"]:
        # Example file paths (adjust these to your actual files)
        base_dir = "stereo_test_calls"
        files = [
            f for f in os.listdir(base_dir) if f.endswith(".mp3") or f.endswith(".wav")
        ]

        if not files:
            print("No audio files found in stereo_test_calls directory")
        else:  # Use first 2 files as an example
            sample_files = files[:2]
            local_file_paths = [
                os.path.abspath(os.path.join(base_dir, f)) for f in sample_files
            ]
            print(f"\nFound {len(local_file_paths)} local files:")
            for path in local_file_paths:
                print(f" - {path}")
            file_paths.extend(local_file_paths)

    # Process URLs
    if choice in ["2", "3"]:
        # Example URLs - replace these with your actual MP3 file URLs
        mp3_urls = [
            "https://raw.githubusercontent.com/Plyraa/warden/refs/heads/main/stereo_test_calls/243801406824559add7684.37683750.mp3",
            "https://raw.githubusercontent.com/Plyraa/warden/refs/heads/main/stereo_test_calls/16952962156823ffe2a06954.22932860.mp3",
        ]

        print(f"\nAdding {len(mp3_urls)} URL references:")
        for url in mp3_urls:
            print(f" - {url}")
            file_paths.append(
                url
            )  # Add URLs directly - they will be processed by the server

    if not file_paths:
        print("No files or URLs to process")
        return

    print(f"\nAnalyzing {len(file_paths)} files:")
    for path in file_paths:
        print(f" - {path}")
    # Send request and get results
    results = analyze_files(file_paths)

    if results:
        print("\n===== RESULTS =====")
        for i, result in enumerate(results):
            print(f"\n=== File {i + 1}: {result['filename']} ===")
            print(f"File Path: {result['file_path']}")

            # Pretty print the complete metric object
            # print("\n=== Complete Metric Object ===")
            # print(json.dumps(result, indent=2, sort_keys=True))
            # print("-" * 50)

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
            print(f"Talk Ratio (Agent/User): {result['talk_ratio']:.2f}")
            print(f"Average Pitch: {result['average_pitch']:.2f} Hz")
            print(f"Words Per Minute: {result['words_per_minute']:.2f} WPM")

            # Display latency points (first 5 only)
            print("\nLatency Points (first 5):")
            for point in result["latency_points"][:5]:
                print(f" - {point['latency_ms']:.2f} ms at {point['moment']:.2f}s")

            if len(result["latency_points"]) > 5:
                print(f" ... and {len(result['latency_points']) - 5} more points")

            print("-" * 50)


if __name__ == "__main__":
    main()

# This script demonstrates how to use the Warden batch API endpoint to analyze audio files.
