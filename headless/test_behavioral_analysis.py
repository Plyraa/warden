#!/usr/bin/env python3
"""
Test script for the new behavioral analysis features
"""

import os
import sys
import traceback
from pathlib import Path

# Add the headless directory to path
sys.path.append(os.path.dirname(__file__))

from gemini_behavioral_analyzer import GeminiBehavioralAnalyzer
from audio_processor import AudioProcessor

def test_behavioral_analyzer():
    """Test the Gemini behavioral analyzer"""
    print("🧪 Testing Gemini Behavioral Analyzer")
    print("=" * 50)
    
    try:
        analyzer = GeminiBehavioralAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test with a sample audio file
        test_files = [
            r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls\test1.mp3",
            r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls\66057328368247d623d0b87.67876133.mp3"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n🎵 Testing with: {os.path.basename(test_file)}")
                result = analyzer.analyze_audio_file(test_file)
                
                if result:
                    print("✅ Analysis successful!")
                    print(f"  - User Churn Risk: {result.userChurnRisk}")
                    if result.userChurnReasoning:
                        print(f"  - Churn Reasoning: {result.userChurnReasoning}")
                    print(f"  - User Repetition: {result.userRepetition}")
                    print(f"  - Agent Repetition: {result.agentRepetition}")
                    break
                else:
                    print("❌ Analysis failed")
            else:
                print(f"⚠️ Test file not found: {test_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Stack trace: {traceback.format_exc()}")

def test_audio_processor_with_behavioral():
    """Test the audio processor with behavioral analysis enabled"""
    print("\n🧪 Testing Audio Processor with Behavioral Analysis")
    print("=" * 50)
    
    try:
        # Create audio processor with behavioral analysis enabled
        audio_dir = Path("audio_downloads")
        audio_dir.mkdir(exist_ok=True)
        
        processor = AudioProcessor(audio_dir=audio_dir, enable_behavioral_analysis=True)
        print("✅ Audio processor with behavioral analysis initialized")
        
        # Test with a sample audio file
        test_file = r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls\test1.mp3"
        
        if os.path.exists(test_file):
            print(f"\n🎵 Processing: {os.path.basename(test_file)}")
            metrics = processor.process_file(test_file)
            
            print("✅ Processing complete!")
            print(f"  - Has behavioral results: {metrics.get('userChurnRisk') is not None}")
            if metrics.get('userChurnRisk') is not None:
                print(f"  - User Churn Risk: {metrics.get('userChurnRisk')}")
                print(f"  - User Repetition: {metrics.get('userRepetition')}")
                print(f"  - Agent Repetition: {metrics.get('agentRepetition')}")
        else:
            print(f"⚠️ Test file not found: {test_file}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Stack trace: {traceback.format_exc()}")

def test_fast_csv_logger():
    """Test the fast CSV logger"""
    print("\n🧪 Testing Fast CSV Logger")
    print("=" * 50)
    
    try:
        from fast_behavioral_csv_logger import get_audio_files_from_csv, process_file_behavioral_analysis, GeminiBehavioralAnalyzer
        
        # Test getting files from directory
        audio_dir = r"C:\Users\Plyra\Desktop\Plyra\jotform\warden\stereo_test_calls"
        if os.path.exists(audio_dir):
            import glob
            test_files = glob.glob(os.path.join(audio_dir, "*.mp3"))[:1]  # Just test one file
            
            if test_files:
                print(f"📁 Found {len(test_files)} test files")
                
                analyzer = GeminiBehavioralAnalyzer()
                result = process_file_behavioral_analysis(analyzer, test_files[0])
                
                print("✅ Fast CSV logger test complete!")
                print(f"  - Status: {result['status']}")
                if result['status'] == 'success':
                    print(f"  - Churn Risk: {result['userChurnRisk']}")
                    print(f"  - Processing Time: {result['processing_time_seconds']}s")
                else:
                    print(f"  - Error: {result['error_message']}")
            else:
                print("⚠️ No test files found")
        else:
            print(f"⚠️ Test directory not found: {audio_dir}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Stack trace: {traceback.format_exc()}")

def main():
    """Run all tests"""
    print("🚀 Running Behavioral Analysis Tests")
    print("=" * 60)
    
    # Test 1: Basic behavioral analyzer
    test_behavioral_analyzer()
    
    # Test 2: Audio processor integration
    test_audio_processor_with_behavioral()
    
    # Test 3: Fast CSV logger
    test_fast_csv_logger()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
