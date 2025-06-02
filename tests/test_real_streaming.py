import requests
import json
import time

def test_real_streaming():
    """Test the streaming endpoint with proper streaming client"""
    
    # Test files from your stereo_test_calls directory
    test_files = [
        "test1.mp3",
        "16952962156823ffe2a06954.22932860.mp3",
        "66057328368247d623d0b87.67876133.mp3",
        "2097202462683d9152e1d734.24483919.mp3",
        "243801406824559add7684.37683750.mp3"
    ]
    
    print(f"Testing REAL streaming with {len(test_files)} files...")
    print("=" * 60)
    print("Each result should appear immediately after processing completes.")
    print("(NOT waiting for all files to finish)")
    print("-" * 60)
    
    start_time = time.time()
    result_count = 0
    
    try:
        # Make streaming request with stream=True
        response = requests.post(
            "http://localhost:8000/batch-stream",
            json={"file_paths": test_files},
            stream=True,  # This enables real streaming
            timeout=300   # 5 minute timeout
        )
        
        if response.status_code == 200:
            print("🔄 Streaming started... waiting for results...")
            print("-" * 40)
            
            # Process each line as it arrives in real-time
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Skip empty lines
                    try:
                        elapsed = time.time() - start_time
                        result = json.loads(line)
                        result_count += 1
                        
                        filename = result.get("filename", "unknown")
                        status = result.get("status", "unknown")
                        
                        print(f"⚡ [{elapsed:.1f}s] Result {result_count}: {filename}")
                        print(f"   Status: {status}")
                        
                        if status == "success":
                            avg_latency = result.get("average_latency", 0)
                            print(f"   Average Latency: {avg_latency:.2f} ms")
                        elif status == "error":
                            error_msg = result.get("error_message", "Unknown error")
                            print(f"   ❌ Error: {error_msg}")
                        
                        print()  # Empty line for readability
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ Failed to parse JSON: {e}")
                        print(f"Raw line: {line}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request: {e}")
        print("Make sure the FastAPI server is running on http://localhost:8000")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print("✅ Streaming test complete!")
    print(f"📊 Results received: {result_count}/{len(test_files)}")
    print(f"⏱️  Total time: {total_time:.1f}s")

if __name__ == "__main__":
    test_real_streaming()
