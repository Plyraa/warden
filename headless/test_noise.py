#!/usr/bin/env python3
"""
Test script for headless noise detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from noise_detection import detect_noise

def test_noise_detection():
    """Test the noise detection functionality"""
    print("🔊 Testing Noise Detection in Headless Mode")
    print("=" * 50)
    
    # Test if the module can be imported and function called
    try:
        # This will test with a non-existent file to check error handling
        result = detect_noise("non_existent_file.wav")
        print("✅ Noise detection module imported and callable")
        print(f"📋 Test result structure: {list(result.keys())}")
        
        if "error" in result and result["error"]:
            print(f"✅ Error handling working: {result['error']}")
        
        # Test expected output format
        expected_keys = {"file", "hasNoise", "noiseInterrupt", "error"}
        actual_keys = set(result.keys())
        
        if expected_keys == actual_keys:
            print("✅ Output format matches expected schema")
        else:
            print(f"⚠️  Output format mismatch:")
            print(f"   Expected: {expected_keys}")
            print(f"   Actual: {actual_keys}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_noise_detection()
