#!/usr/bin/env python3
"""
Test script for Warden Headless API
Tests all endpoints with different scenarios
"""

import requests
import json
import time
import sys
import os
from pathlib import Path

# API Configuration
API_BASE = "http://localhost:8030"
TEST_FILES_DIR = r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/isAlive")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing endpoint"""
    print("\n📊 Testing Batch Processing...")
    
    # Test with full paths to local files
    test_files = [
        os.path.join(TEST_FILES_DIR, "18122608606843567790f6a3.38069641.mp3"),
        os.path.join(TEST_FILES_DIR, "207978285268425581ee85e6.86799284.mp3")
    ]
    
    payload = {
        "file_paths": test_files
    }
    
    try:
        print(f"Sending request with files: {[os.path.basename(f) for f in test_files]}")
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            response_data = response.json()
            results = response_data.get('results', [])
            print(f"✅ Successfully processed {len(results)} files")
            
            for i, result in enumerate(results):
                print(f"\n--- File {i+1}: {result['filename']} ---")
                print(f"Status: {result['status']}")
                if result['status'] == 'success':
                    print(f"Average Latency: {result['average_latency']:.2f}ms")
                    print(f"P50 Latency: {result['p50_latency']:.2f}ms")
                    print(f"P90 Latency: {result['p90_latency']:.2f}ms")
                    print(f"Talk Ratio: {result['talk_ratio']:.2f}")
                    print(f"AI Interrupting User: {result['ai_interrupting_user']}")
                    print(f"User Interrupting AI: {result['user_interrupting_ai']}")
                    print(f"Overlap Counts - AI->User: {result['ai_user_overlap_count']}, User->AI: {result['user_ai_overlap_count']}")
                    print(f"Average Pitch: {result['average_pitch']:.2f}Hz")
                    print(f"Words per Minute: {result['words_per_minute']:.2f}")
                    print(f"Latency Points: {len(result['latency_points'])}")
                else:
                    print(f"Error: {result['error_message']}")
            
            return True
        else:
            print(f"❌ Batch processing failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False

def test_streaming_processing():
    """Test streaming batch processing endpoint - shows real-time results"""
    print("\n🌊 Testing Streaming Batch Processing...")
    
    # Use URLs for longer processing to better demonstrate streaming
    test_files = [
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/66057328368247d623d0b87.67876133.mp3",
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/17457745626800ad609b0bd7.58327851.mp3"
    ]
    
    payload = {
        "file_paths": test_files
    }
    
    try:
        print(f"Streaming request with {len(test_files)} files...")
        print("⏳ Results will appear as each file completes processing...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/batch-stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,  # Enable streaming
            timeout=300
        )
        
        if response.status_code == 200:
            print("✅ Stream started successfully")
            print("📡 Streaming results:")
            print("-" * 80)
            
            result_count = 0
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Skip empty lines
                    result_count += 1
                    elapsed = time.time() - start_time
                    
                    try:
                        result = json.loads(line)
                        filename = result.get('filename', 'Unknown')
                        status = result.get('status', 'Unknown')
                        
                        print(f"🎯 Result #{result_count} ({elapsed:.1f}s): {filename}")
                        print(f"   Status: {status}")
                        
                        if status == "success":
                            avg_latency = result.get('average_latency', 0)
                            talk_ratio = result.get('talk_ratio', 0)
                            print(f"   Avg Latency: {avg_latency:.1f}ms")
                            print(f"   Talk Ratio: {talk_ratio:.2f}")
                        else:
                            error_msg = result.get('error_message', 'Unknown error')
                            print(f"   Error: {error_msg}")
                        
                        print("-" * 40)
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error: {e}")
                        print(f"   Raw line: {line}")
            
            total_time = time.time() - start_time
            print(f"🏁 Streaming completed in {total_time:.2f} seconds")
            print(f"📊 Processed {result_count} files")
            return True
            
        else:
            print(f"❌ Streaming failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid files"""
    print("\n⚠️  Testing Error Handling...")
    
    # Test with non-existent files using full paths
    payload = {
        "file_paths": [
            os.path.join(TEST_FILES_DIR, "nonexistent.mp3"),
            os.path.join(TEST_FILES_DIR, "another_fake.wav")
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/batch",
            json=payload,
            headers={"Content-Type": "application/json"}        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            results = response_data.get('results', [])
            print(f"Processed {len(results)} files (with errors expected)")
            
            for result in results:
                print(f"File: {result['filename']}, Status: {result['status']}")
                if result['status'] == 'error':
                    print(f"  Error: {result['error_message']}")
            
            return True
        else:
            print(f"❌ Error handling test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def test_url_processing():
    """Test URL processing (if needed)"""
    print("\n🌐 Testing URL Processing...")
    
    # Test with a sample audio URL (you can replace with actual URL)
    payload = {
        "file_paths": ["https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/66057328368247d623d0b87.67876133.mp3",
        "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/17457745626800ad609b0bd7.58327851.mp3"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"URL Processing Status: {response.status_code}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ URL processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Warden Headless API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Batch Processing", test_batch_processing),
        ("Streaming Processing", test_streaming_processing),
        ("Error Handling", test_error_handling),
        ("URL Processing", test_url_processing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Warden Headless is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
